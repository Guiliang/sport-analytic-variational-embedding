import datetime
import json
import numpy as np
import scipy.io as sio
from config.LSTM_diff_config import LSTMDiffCongfig
from config.cvrnn_config import CVRNNCongfig
from support.data_processing_tools import read_feature_within_events, read_features_within_events
from support.model_tools import get_model_and_log_name, get_data_name
from support.plot_tools import plot_diff


def compute_next_goal_calibration_values(actions_all, home_away):
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


def read_results_values(result_dir):
    with open(result_dir, 'rb') as f:
        lines = f.readlines()
    results = []
    times = []
    for line in lines[2:]:
        times.append(float(line.split('time')[1].split('is')[0]))
        results.append(float(line.split('is')[1]))
    return times, results


def obtain_model_predictions(model_data_store_dir, directory, data_name):
    # 'model_2101_three_cut_cvrnn_accumu_Qs_featureV1_latent128_x83_y150_batch32_iterate30_lr0.0001_normal_MaxTL10_LSTM512_box'
    with open(model_data_store_dir + "/" + directory + "/" + data_name) as outfile:
        model_output = json.load(outfile)
    model_values_all = []
    for i in range(0, len(model_output)):
        home_value = model_output[str(i)]['home']
        away_value = model_output[str(i)]['away']
        end_value = model_output[str(i)]['end']
        model_values_all.append([home_value, away_value, end_value])
    return np.asarray(model_values_all)


def validate_next_goal(model_data_store_dir,
                       testing_dir_games_all,
                       data_name,
                       data_store_dir,
                       file_writer=None):
    real_label_record = np.ones([len(testing_dir_games_all), 5000, 3]) * -100
    output_values_record = np.ones([len(testing_dir_games_all), 5000, 3]) * -100
    output_labels_record = np.ones([len(testing_dir_games_all), 5000, 3]) * -100

    event_diff = []
    event_acc = []
    event_numbers = []

    for dir_index in range(0, len(testing_dir_games_all)):
        print('Processing game {0}'.format(dir_index))
        testing_dir = testing_dir_games_all[dir_index]

        home_away_identifier_name = 'home_away_identifier_game_{0}-playsequence-wpoi.mat'.format(str(testing_dir))
        home_away_identifier = sio.loadmat(
            model_data_store_dir + "/" + str(testing_dir) + "/" + home_away_identifier_name)
        home_away = home_away_identifier['home_away'][0]
        # model_values = [[1, 0, 0]] * 1519  # TODO: test
        actions_all = read_features_within_events(feature_name_list=['name'],
                                                  data_path=data_store_dir,
                                                  directory=str(testing_dir) + '-playsequence-wpoi.json')
        target_values = np.asarray(compute_next_goal_calibration_values(actions_all, home_away))
        real_label_all = target_values
        if data_name is None:
            model_values = None
            model_values_argmax = None
        else:
            model_values = obtain_model_predictions(model_data_store_dir, testing_dir, data_name)
            temp = np.argmax(model_values, axis=1)
            model_values_argmax = np.zeros([model_values.shape[0], model_values.shape[1]])
            model_values_argmax[np.arange(model_values.shape[0]), temp] = 1
            # output_label_all = np.abs(model_values - target_values[:len(model_values)])

        real_label_record[dir_index][:len(real_label_all)] = real_label_all
        output_values_record[dir_index][:len(model_values)] = model_values
        output_labels_record[dir_index][:len(model_values)] = model_values_argmax

    for i in range(0, output_values_record.shape[1]):
        real_outcome_record_step = real_label_record[:, i, :]
        model_output_values_record_step = output_values_record[:, i, :]
        model_output_labels_record_step = output_labels_record[:, i, :]
        diff_sum = 0
        acc_correct_num = 0
        total_number = 0
        print_flag = True
        include_flag = True
        for index in range(0, len(real_outcome_record_step)):
            if model_output_values_record_step[index][0] == -100 or real_outcome_record_step[index][0] == -100:
                include_flag = False
                continue
            diff = abs(model_output_values_record_step[index] - real_outcome_record_step[index])
            diff_sum += diff

            if np.array_equal(model_output_labels_record_step[index], real_outcome_record_step[index]):
                acc_correct_num += 1
            total_number += 1
        if not include_flag:
            continue
        if total_number == 0:
            continue
        event_diff.append(np.divide(diff_sum, total_number))
        event_acc.append(float(acc_correct_num) / total_number)
        event_numbers.append(i)

        # if file_writer is not None:
        #     file_writer.write('diff of time {0} is {1}\n'.format(str(i), str(float(diff_sum) / total_number)))

        if print_flag:
            if i % 100 == 0 and total_number > 0:
                print('diff of time {0} is {1}, acc is {2}'.format(str(i),
                                                                   str(np.divide(diff_sum, total_number)),
                                                                   str(float(acc_correct_num) / total_number)))
    return np.asarray(event_diff), event_acc, event_numbers


if __name__ == '__main__':
    # cvrnn_result_dir = '../interface/cvrnn_testing_results2019September23.txt'
    # game_time, cvrnn_results = read_results_values(cvrnn_result_dir)
    # plot_diff(game_time, cvrnn_results)

    validated_model_type = [
        {'model_category': 'cvrnn', 'model_number': '1801', 'player_info': '_box'},
        {'model_category': 'lstm_Qs', 'model_number': '901', 'player_info': ''},

    ]

    model_category_msg_all = []
    event_numbers_all = []
    acc_all = []
    diff_all = []
    for model_info_setting in validated_model_type:

        model_category = model_info_setting.get('model_category')
        model_number = model_info_setting.get('model_number')
        player_info = model_info_setting.get('player_info')
        if model_category == 'cvrnn':
            predicted_target = '_PlayerLocalId'  # playerId_
            icehockey_cvrnn_config_path = "../../environment_settings/icehockey_cvrnn{0}_config{1}.yaml". \
                format(predicted_target, player_info)
            icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
            # testing_file = open('./cvrnn{1}_model{2}_testing_results{0}.txt'. \
            #                     format(datetime.date.today().strftime("%Y%B%d"), player_info, str(model_number)),
            #                     'wb')
        elif model_category == 'lstm_Qs':
            icehockey_config_path = "../../environment_settings/ice_hockey_predict_Qs_lstm{0}.yaml" \
                .format(player_info)
            icehockey_model_config = LSTMDiffCongfig.load(icehockey_config_path)
            # testing_file = open('./LSTM_diff{1}_model{2}_testing_results{0}.txt'. \
            #                     format(datetime.date.today().strftime("%Y%B%d"), '', str(model_number)), 'wb')
        elif model_category == 'zero':
            icehockey_model_config = None
        else:
            raise ValueError("uknown model catagoery {0}".format(model_category))

        # if icehockey_model_config is not None:
        #     saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
        #                                                         model_catagoery=model_category)
        testing_dir_games_all = []
        #
        # if 'v2' not in player_info and model_category == 'cvrnn':
        #     saved_network_dir = saved_network_dir.replace('cvrnn_saved_networks', 'bak_cvrnn_saved_networks')
        # if 'v2' not in player_info and model_category == 'lstm_diff':
        #     saved_network_dir = saved_network_dir.replace('lstm_saved_networks', 'bak_lstm_saved_networks')

        with open('../../sport_resource/ice_hockey_201819/' + '/testing_file_dirs_all_v2.csv', 'rb') as f:
            # with open(saved_network_dir + '/testing_file_dirs_all.csv', 'rb') as f:
            testing_dir_all = f.readlines()
        for testing_dir in testing_dir_all:
            testing_dir_games_all.append(str(int(testing_dir)))
        model_data_store_dir = "/Local-Scratch/oschulte/Galen/Ice-hockey-data/2018-2019/"
        data_store = '/Local-Scratch/oschulte/Galen/2018-2019/'
        if model_category != 'zero':
            data_name = get_data_name(icehockey_model_config, model_category, model_number)
            data_name = data_name.replace('Qs', 'next_Qs')
        else:
            data_name = None

        print(model_category + '_' + model_number + player_info)

        diff, acc, event_numbers = validate_next_goal(model_data_store_dir,
                                                      testing_dir_games_all,
                                                      data_name,
                                                      data_store,
                                                      None)
        event_numbers_all.append(event_numbers)
        acc_all.append(acc)
        diff_all.append(diff)
        model_category_msg_all.append(model_category + '_' + model_number + player_info)
        # model_category_msg_all.append(model_category + player_info)
        # testing_file.close()

    plot_diff(game_time_list=event_numbers_all, diff_values_list=acc_all,
              model_category_all=model_category_msg_all)
