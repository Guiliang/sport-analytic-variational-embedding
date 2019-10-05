import numpy as np
import os
import json
import math
import datetime
import scipy.io as sio
from support.data_processing_tools import read_features_within_events, read_feature_within_events
from support.model_tools import get_data_name


class Calibration:
    def __init__(self, bins, source_data_dir, calibration_features,
                 config, model_data_store_dir,
                 apply_old, apply_difference, model_type,
                 model_number, player_info, calibration_type,
                 testing_dir_all,
                 focus_actions_list=[]):
        self.calibration_type = calibration_type
        self.bins = bins
        # self.bins_names = bins.keys()
        self.apply_old = apply_old
        self.apply_difference = apply_difference
        self.data_path = source_data_dir
        self.calibration_features = calibration_features
        if self.apply_difference:
            self.calibration_values_all_dict = {'all': {'cali_sum': [0], 'model_sum': [0], 'number': 0}}
        else:
            self.calibration_values_all_dict = {'all': {'cali_sum': [0, 0, 0], 'model_sum': [0, 0, 0], 'number': 0}}
        self.data_store_dir = model_data_store_dir
        self.config = config
        self.focus_actions_list = focus_actions_list
        if self.apply_difference:
            self.save_calibration_dir = './calibration_results/difference-calibration_{2}_{0}_{1}{3}_model{4}.txt'. \
                format(str(self.focus_actions_list), datetime.date.today().strftime("%Y%B%d"), model_type,
                       player_info, model_number)
        else:
            self.save_calibration_dir = './calibration_results/calibration_{2}_{0}_{1}{3}_model{4}.txt'. \
                format(str(self.focus_actions_list), datetime.date.today().strftime("%Y%B%d"), model_type,
                       player_info, model_number)
        self.save_calibration_file = open(self.save_calibration_dir, 'w')
        if apply_difference:
            self.teams = ['home-away']
        else:
            if self.calibration_type == 'next_goal':
                self.teams = ['home', 'away', 'end']
            elif self.calibration_type == 'score_diff':
                self.teams = ['home', 'away']
            # learning_rate = tt_lstm_config.learn.learning_rate
            # pass
        data_name = get_data_name(config=config, model_catagoery=model_type, model_number=model_number)

        if self.calibration_type == 'next_goal':
            self.data_name = data_name.replace('Qs', 'next_Qs')
        elif self.calibration_type == 'score_diff':
            self.data_name = data_name.replace('Qs', 'accumu_Qs')
        else:
            raise ValueError('unknown calibration type {0}'.format(self.calibration_type))
        print(self.data_name)
        self.calibration_type = calibration_type
        self.testing_dir_all = testing_dir_all

    def __del__(self):
        print 'ending bak_calibration'
        print self.save_calibration_file.close()

    def recursive2construct(self, store_dict_str, depth):
        feature_number = len(self.calibration_features)
        if depth >= feature_number:
            if self.apply_difference:
                self.calibration_values_all_dict.update({store_dict_str: {'cali_sum': [0],
                                                                          'model_sum': [0],
                                                                          'number': 0}})
            else:
                self.calibration_values_all_dict.update({store_dict_str: {'cali_sum': [0, 0, 0],
                                                                          'model_sum': [0, 0, 0],
                                                                          'number': 0}})
            return
        calibration_feature = self.calibration_features[depth]
        feature_range = self.bins.get(calibration_feature).get('range')
        for value in feature_range:
            # store_dict_str = '-' + store_dict_str if len(store_dict_str) > 0 else store_dict_str
            store_dict_str_update = store_dict_str + calibration_feature + '_' + str(value) + '-'
            self.recursive2construct(store_dict_str_update, depth + 1)

    def construct_bin_dicts(self):
        """create bak_calibration dict"""
        self.recursive2construct('', 0)

    def compute_next_goal_calibration_values(self, actions_all, home_away):
        """ground truth value for each game"""
        pre_index = 0
        cali_home = [0] * len(actions_all)
        cali_away = [0] * len(actions_all)
        cali_end = [0] * len(actions_all)
        for index in range(0, len(actions_all)):
            action = actions_all[index]
            if action['name'] == 'goal':
                if home_away[index] == 1:
                    cali_home[pre_index:index] = [1] * (index - pre_index)
                elif home_away[index] == 0:
                    cali_away[pre_index:index] = [1] * (index - pre_index)
                pre_index = index
            if index == len(actions_all) - 1:
                cali_end[pre_index:index] = [1] * (index - pre_index)
        return zip(cali_home, cali_away, cali_end)

    def compute_score_diff_calibration_values(self, home_final_goal, away_final_goal, home_away):
        """ground truth value for each game"""
        cali_home = [home_final_goal] * len(home_away)
        cali_away = [away_final_goal] * len(home_away)
        return zip(cali_home, cali_away)

    def compute_score_diff_based_values(self, directory, reward):
        base_home_goal = [0] * len(reward)
        base_away_goal = [0] * len(reward)

        for index in range(0, len(reward)):
            if reward[index] == 1:
                for i in range(index + 1, len(reward)):
                    base_home_goal[i] += 1
            if reward[index] == -1:
                for i in range(index + 1, len(reward)):
                    base_away_goal[i] += 1
            pass
        # print(base_home_goal)
        # print (base_away_goal)
        return [base_home_goal, base_away_goal]

    def obtain_model_prediction(self, directory):
        """model predicted value for each game"""
        # directory = '16198'
        print(self.data_store_dir + "/" + directory + "/" + self.data_name)

        with open(self.data_store_dir + "/" + directory + "/" + self.data_name) as outfile:
            model_output = json.load(outfile)
        "model_three_cut_featureV1_latent128_x76_y150_batch32_iterate30_lr0.0001_normal_MaxTL10_LSTM512"
        return model_output

    def aggregate_calibration_values(self):
        """update bak_calibration dict by each game"""
        dir_all = os.listdir(self.data_path)
        # dir_all = ['919069.json']  # TODO: test
        # self.data_path = '/Users/liu/Desktop/'
        # for json_dir in dir_all:
        for json_dir in self.testing_dir_all:
            features_all = []
            for calibration_feature in self.calibration_features:
                features = self.bins.get(calibration_feature).get('feature_name')
                if isinstance(features, str):
                    features_all.append(features)
                else:
                    for feature in features:
                        features_all.append(feature)

            model_values = self.obtain_model_prediction(directory=json_dir.split('-')[0])
            game_files = os.listdir(self.data_store_dir + "/" + json_dir.split('-')[0])
            for filename in game_files:
                if 'home_away' in filename:
                    home_away_identifier_name = filename

            home_away_identifier = sio.loadmat(
                self.data_store_dir + "/" + json_dir.split('-')[0] + "/" + home_away_identifier_name)
            home_away = home_away_identifier['home_away'][0]
            # model_values = [[1, 0, 0]] * 1519  # TODO: test
            actions_all = read_features_within_events(feature_name_list=['name'],
                                                      data_path=self.data_path, directory=json_dir)
            if self.calibration_type == 'next_goal':

                calibration_values = self.compute_next_goal_calibration_values(actions_all, home_away)
            elif self.calibration_type == 'score_diff':
                reward = sio.loadmat(
                    self.data_store_dir + "/" + json_dir.split('-')[0] + "/"
                    + 'reward_{0}-playsequence-wpoi'.format(json_dir.split('-')[0]))
                reward = reward['reward'][0]
                home_final_goal = 0
                away_final_goal = 0
                for r in reward:
                    if r == 1:
                        home_final_goal += 1
                    if r == -1:
                        away_final_goal += 1

                base_goals = self.compute_score_diff_based_values(directory=json_dir.split('-')[0],
                                                                  reward=reward)
                calibration_values = self.compute_score_diff_calibration_values(home_final_goal=home_final_goal,
                                                                                away_final_goal=away_final_goal,
                                                                                home_away=home_away)

            else:
                raise ValueError('unknown calibration type {0}'.format(self.calibration_type))

            features_values_dict_all = read_features_within_events(feature_name_list=features_all,
                                                                   data_path=self.data_path,
                                                                   directory=json_dir)
            features_values_dict_all_new = []
            for feature_values_index in range(0, len(features_values_dict_all)):
                feature_values_dict = features_values_dict_all[feature_values_index]
                feature_values_dict.update({'home_away': home_away[feature_values_index]})
                features_values_dict_all_new.append(feature_values_dict)

            for index in range(0, len(model_values)):
                action = actions_all[index]['name']  # find the action we focus
                continue_flag = False if len(self.focus_actions_list) == 0 else True
                for f_action in self.focus_actions_list:
                    if f_action in action:
                        # print action
                        continue_flag = False
                if continue_flag:
                    continue

                features_values_dict = features_values_dict_all_new[index]
                cali_dict_str = ''
                for calibration_feature in self.calibration_features:
                    if calibration_feature == 'period':
                        value = features_values_dict.get('period')
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + str(value) + '-'
                    elif calibration_feature == 'scoreDifferential':
                        value = features_values_dict.get('scoreDifferential')
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + str(value) + '-'
                    elif calibration_feature == 'zone':
                        value = features_values_dict.get('zone')
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + value + '-'
                    elif calibration_feature == 'manpowerSituation':
                        value = features_values_dict.get('manpowerSituation')
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + str(value) + '-'
                    elif calibration_feature == 'home_away':
                        value = features_values_dict.get('home_away')
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + str(value) + '-'
                    else:
                        raise ValueError('unknown feature ' + calibration_feature)

                calibration_value = calibration_values[index]
                model_value = model_values[str(index)]

                cali_bin_info = self.calibration_values_all_dict.get(cali_dict_str)
                # print cali_dict_str
                assert cali_bin_info is not None
                cali_sum = cali_bin_info.get('cali_sum')
                model_sum = cali_bin_info.get('model_sum')
                number = cali_bin_info.get('number')
                number += 1
                if self.apply_difference:
                    if self.calibration_type == 'next_goal':
                        cali_sum[0] = cali_sum[0] + (calibration_value[0] - calibration_value[1])
                        model_sum[0] = model_sum[0] + (model_value['home'] - model_value['away'])
                    elif self.calibration_type == 'score_diff':
                        cali_sum[0] = cali_sum[0] + (calibration_value[0] + base_goals[0][index] -
                                                     calibration_value[1] - base_goals[1][index])
                        model_sum[0] = model_sum[0] + (model_value['home'] - model_value['away'])
                else:
                    for i in range(len(self.teams)):  # [home, away,end]
                        cali_sum[i] = cali_sum[i] + calibration_value[i]

                        if self.calibration_type == 'next_goal':
                            model_sum[i] = model_sum[i] + model_value[self.teams[i]]
                        elif self.calibration_type == 'score_diff':
                            model_sum[i] = model_sum[i] + model_value[self.teams[i]] + base_goals[i][index]
                self.calibration_values_all_dict.update({cali_dict_str: {'cali_sum': cali_sum,
                                                                         'model_sum': model_sum,
                                                                         'number': number}})

                cali_bin_info = self.calibration_values_all_dict.get('all')
                cali_sum = cali_bin_info.get('cali_sum')
                model_sum = cali_bin_info.get('model_sum')
                number = cali_bin_info.get('number')
                number += 1
                if self.apply_difference:
                    cali_sum[0] = cali_sum[0] + (calibration_value[0] - calibration_value[1])
                    model_sum[0] = model_sum[0] + (model_value['home'] - model_value['away'])
                else:
                    for i in range(len(self.teams)):  # [home, away (, +end)]
                        cali_sum[i] = cali_sum[i] + calibration_value[i]
                        model_sum[i] = model_sum[i] + model_value[self.teams[i]]

                self.calibration_values_all_dict.update({'all': {'cali_sum': cali_sum,
                                                                 'model_sum': model_sum,
                                                                 'number': number}})

                # break

    def compute_distance(self):
        cali_dict_strs = self.calibration_values_all_dict.keys()
        for cali_dict_str in cali_dict_strs:
            cali_bin_info = self.calibration_values_all_dict.get(cali_dict_str)
            kld_sum = 0
            mae_sum = 0
            if cali_bin_info['number'] == 0:
                print "number of bin {0} is 0".format(cali_dict_str)
                continue
            cali_record_dict = 'Bin:' + cali_dict_str
            for i in range(len(self.teams)):  # [home, away,end]
                cali_prob = float(cali_bin_info['cali_sum'][i]) / cali_bin_info['number']
                model_prob = float(cali_bin_info['model_sum'][i]) / cali_bin_info['number']
                cali_record_dict += '\t{0}_number'.format(self.teams[i]) + ":" + str(cali_bin_info['number'])
                cali_record_dict += '\t{0}_cali'.format(self.teams[i]) + ":" + str(cali_prob)
                cali_record_dict += '\t{0}_model'.format(self.teams[i]) + ":" + str(model_prob)
                model_prob = model_prob + 1e-10
                cali_prob = cali_prob + 1e-10
                try:
                    kld = cali_prob * math.log(cali_prob / model_prob)
                except:
                    print 'kld is ' + str(cali_prob / model_prob)
                    kld = 0
                kld_sum += kld
                ae = abs(cali_prob - model_prob)
                mae_sum = mae_sum + ae
            cali_record_dict += '\tkld:' + str(kld_sum)
            cali_record_dict += '\tmae:' + str(float(mae_sum) / len(self.teams))
            self.save_calibration_file.write(str(cali_record_dict) + '\n')
