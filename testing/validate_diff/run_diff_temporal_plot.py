import sys
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
print sys.path
import datetime
import json
import numpy as np
import matplotlib
from matplotlib.pyplot import cm
from config.LSTM_diff_config import LSTMDiffCongfig
from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.stats_encoder_config import EncoderConfig
from support.data_processing_tools import read_feature_within_events
from support.model_tools import get_model_and_log_name, get_data_name
from support.plot_tools import plot_diff, plot_cv_diff

matplotlib.use('Agg')


def read_results_values(result_dir):
    with open(result_dir, 'rb') as f:
        lines = f.readlines()
    results = []
    times = []
    for line in lines[2:]:
        times.append(float(line.split('time')[1].split('is')[0]))
        results.append(float(line.split('is')[1]))
    return times, results


def obtain_model_predictions(model_data_store_dir, directory, data_name, running_number=None):
    # 'model_2101_three_cut_cvrnn_accumu_Qs_featureV1_latent128_x83_y150_batch32_iterate30_lr0.0001_normal_MaxTL10_LSTM512_box'
    with open(model_data_store_dir + "/" + directory + "/" + data_name) as outfile:
        model_output = json.load(outfile)

    if running_number is not None:
        model_values_all = []
        try:
            for i in range(0, len(model_output[str(running_number)])):
                home_value = model_output[str(running_number)][str(i)]['home']
                away_value = model_output[str(running_number)][str(i)]['away']
                end_value = model_output[str(running_number)][str(i)]['end']
                model_values_all.append([home_value, away_value, end_value])
            model_values_all = np.asarray(model_values_all)
        except:
            print(data_name)
        return model_values_all

    else:
        model_values_all = []
        for i in range(0, len(model_output)):
            home_value = model_output[str(i)]['home']
            away_value = model_output[str(i)]['away']
            end_value = model_output[str(i)]['end']
            model_values_all.append([home_value, away_value, end_value])
        return np.asarray(model_values_all)


def validate_score_diff(model_data_store_dir,
                        data_name,
                        source_data_dir,
                        data_store,
                        model_category,
                        file_writer=None,
                        cv_number=None):
    length_max = 5000
    length_min = 5000

    real_label_record_all = None
    output_label_record_all = None
    game_time_record_all = None

    for running_number in range(0, cv_number):
        saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                            model_catagoery=model_category,
                                                            running_number=running_number)
        testing_dir_games_all = []
        # with open('../../sport_resource/ice_hockey_201819/' + '/testing_file_dirs_all_v2.csv', 'rb') as f:
        with open(saved_network_dir + '/testing_file_dirs_all.csv', 'rb') as f:
            testing_dir_all = f.readlines()
        for testing_dir in testing_dir_all:
            testing_dir_games_all.append(str(int(testing_dir)))

        testing_dir_games_all = testing_dir_games_all[:20]
        real_label_record = np.ones([len(testing_dir_games_all), length_max]) * -100
        output_label_record = np.ones([len(testing_dir_games_all), length_max]) * -100
        game_time_record = np.ones([len(testing_dir_games_all), length_max]) * -100

        for dir_index in range(0, len(testing_dir_games_all)):
            print('Processing game {0}'.format(dir_index))
            testing_dir = testing_dir_games_all[dir_index]
            if data_name is not None:
                model_values = obtain_model_predictions(model_data_store_dir, testing_dir, data_name, running_number)

            score_difference_game = read_feature_within_events(testing_dir,
                                                               data_store,
                                                               'scoreDifferential',
                                                               transfer_home_number=True,
                                                               data_store=source_data_dir,
                                                               allow_overtime=False)
            game_time_list = read_feature_within_events(testing_dir,
                                                        data_store,
                                                        'gameTime',
                                                        transfer_home_number=False,
                                                        data_store=source_data_dir,
                                                        allow_overtime=False)

            if data_name is None:
                output_label_all = np.asarray(len(score_difference_game) * [0]) + score_difference_game
                real_label_all = [score_difference_game[-1]] * len(score_difference_game)
                game_time_list = []
                for j in range(0, len(score_difference_game)):  # TODO: how to map to the time under cross-validation?
                    game_time_list.append(float(3600) / len(score_difference_game) * j)
            else:
                real_label_all = [score_difference_game[-1]] * len(score_difference_game)
                output_label_all = model_values[:len(score_difference_game), 0] - \
                                   model_values[:len(score_difference_game), 1] + score_difference_game[
                                                                                  :len(score_difference_game)]

            real_label_record[dir_index][:len(real_label_all)] = real_label_all
            output_label_record[dir_index][:len(output_label_all)] = output_label_all
            game_time_record[dir_index][:len(game_time_list)] = game_time_list
        if real_label_record_all is None:
            real_label_record_all = real_label_record
            output_label_record_all = output_label_record
            game_time_record_all = game_time_record
        else:
            real_label_record_all = np.concatenate([real_label_record_all, real_label_record], axis=0)
            output_label_record_all = np.concatenate([output_label_record_all, output_label_record], axis=0)
            game_time_record_all = np.concatenate([game_time_record_all, game_time_record], axis=0)

    acc_diff_mean_by_event = []
    acc_diff_var_by_event = []
    acc_global = []
    game_time_diff_record_list = []
    game_time_list = []
    for i in range(0, 3601):
        game_time_diff_record_list.append([])
        game_time_list.append(i)
    for i in range(0, length_max):
        include_number = 0
        real_outcome_record_step = real_label_record_all[:, i]
        model_output_record_step = output_label_record_all[:, i]
        game_time_record_step = game_time_record_all[:, i]
        diff_list = []
        total_number = 0
        print_flag = True
        check_flag = False
        include_flag = False
        for win_index in range(0, len(real_outcome_record_step)):
            if model_output_record_step[win_index] == -100 or \
                            real_outcome_record_step[win_index] == -100 or \
                            game_time_record_step[win_index] == -100:
                check_flag = True
                # include_flag = False
                continue
            else:
                include_flag = True

            diff = abs(model_output_record_step[win_index] - real_outcome_record_step[win_index])
            game_time_index = int(game_time_record_step[win_index])
            game_time_diff_record_list[game_time_index].append(diff)
            diff_list.append(diff)
            acc_global.append(diff)
            total_number += 1
        if check_flag:
            diff_list_new = []
            for diff in diff_list:
                if diff < 0.2:
                    diff_list_new.append(diff)
            if len(diff_list_new) == 0:
                include_flag = False

        if include_flag:
            include_number += 1
            acc_diff_mean_by_event.append(np.mean(np.asarray(diff_list)))
            acc_diff_var_by_event.append(np.var(np.asarray(diff_list)))
            if file_writer is not None:
                file_writer.write('diff of event {0} is {1}\n'.format(str(include_number), str(acc_diff_mean_by_event[include_number])))

            if print_flag:
                if include_number % 100 == 0:
                    print('diff of event {0} is {1}'.format(str(include_number), str(acc_diff_mean_by_event[include_number])))
        else:
            continue
        # event_numbers.append(i)

    acc_diff_mean_by_time = []
    acc_diff_var_by_time = []
    for i in range(0, 3601):
        game_time_diff_list = game_time_diff_record_list[i]
        acc_diff_mean_by_time.append(np.mean(np.asarray(game_time_diff_list)))
        acc_diff_var_by_time.append(np.var(np.asarray(game_time_diff_list)))

        if i % 100 == 0:
            print('diff of time {0} is {1}'.format(str(i), str(acc_diff_mean_by_event[i])))

    print('diff of {0} has the mean {1} and variance {2}.'.format(model_category,
                                                                  str(np.mean(np.asarray(acc_global))),
                                                                  str(np.var(np.asarray(acc_global)))))
    return np.asarray(acc_diff_mean_by_event), np.asarray(acc_diff_var_by_event), \
           range(len(acc_diff_mean_by_event)), \
           np.asarray(acc_diff_mean_by_time), np.asarray(acc_diff_var_by_time), game_time_list


if __name__ == '__main__':
    # cvrnn_result_dir = '../interface/cvrnn_testing_results2019September23.txt'
    # game_time, cvrnn_results = read_results_values(cvrnn_result_dir)
    # plot_diff(game_time, cvrnn_results)

    model_data_store_dir = "/Local-Scratch/oschulte/Galen/Ice-hockey-data/2018-2019/"
    data_store = '/Local-Scratch/oschulte/Galen/2018-2019/'
    validated_model_type = [
        # {'model_category': 'lstm_diff', 'model_name': 'N/A', 'model_number': '2101', 'player_info': ''},
        # {'model_category': 'lstm_diff', 'model_name': 'N/A', 'model_number': '1801', 'player_info': ''},
        # {'model_category': 'lstm_diff', 'model_name': 'N/A', 'model_number': '1501', 'player_info': ''},
        # {'model_category': 'lstm_diff', 'model_name': 'N/A', 'model_number': '1201', 'player_info': ''},
        {'model_category': 'lstm_diff', 'model_name': 'N/A', 'model_number': '901', 'player_info': ''},
        {'model_category': 'lstm_diff', 'model_name': 'pid', 'model_number': '2101', 'player_info': '_pid'},
        # {'model_category': 'lstm_diff', 'model_name': 'pid', 'model_number': '1801', 'player_info': '_pid'},
        # {'model_category': 'lstm_diff', 'model_name': 'pid', 'model_number': '1501', 'player_info': '_pid'},
        # {'model_category': 'lstm_diff', 'model_name': 'pid', 'model_number': '1201', 'player_info': '_pid'},
        # {'model_category': 'lstm_diff', 'model_name': 'pid', 'model_number': '901', 'player_info': '_pid'},
        # {'model_category': 'lstm_diff', 'model_name': 'N/A', 'model_number': '2101', 'player_info': ''},
        # {'model_category': 'encoder', 'model_name': 'DE', 'model_number': '1801', 'player_info': ''},
        # {'model_category': 'encoder', 'model_name': 'DE', 'model_number': '1501', 'player_info': ''},
        {'model_category': 'encoder', 'model_name': 'DE', 'model_number': '1201', 'player_info': ''},
        # {'model_category': 'encoder', 'model_name': 'DE', 'model_number': '901', 'player_info': ''},
        {'model_category': 'cvae', 'model_name': 'CVAE', 'model_number': '1801', 'player_info': ''},
        # {'model_category': 'cvae', 'model_name': 'CVAE', 'model_number': '1501', 'player_info': ''},
        # {'model_category': 'cvae', 'model_name': 'CVAE', 'model_number': '1201', 'player_info': ''},
        # {'model_category': 'cvae', 'model_name': 'CVAE', 'model_number': '901', 'player_info': ''},
        {'model_category': 'vhe', 'model_name': 'VHE', 'model_number': '1801', 'player_info': ''},
        # {'model_category': 'vhe', 'model_name': 'VHE', 'model_number': '1501', 'player_info': ''},
        # {'model_category': 'vhe', 'model_name': 'VHE', 'model_number': '1201', 'player_info': ''},
        # {'model_category': 'vhe', 'model_name': 'VHE', 'model_number': '901', 'player_info': ''},
        # {'model_category': 'cvrnn', 'model_name': 'VHER', 'model_number': '1801', 'player_info': ''},
        {'model_category': 'cvrnn', 'model_name': 'VHER', 'model_number': '1501', 'player_info': ''},
        # {'model_category': 'cvrnn', 'model_name': 'VHER', 'model_number': '1201', 'player_info': ''},
        # {'model_category': 'cvrnn', 'model_name': 'VHER', 'model_number': '901', 'player_info': ''},
    ]
    # validated_model_type = [
    #     {'model_category': 'cvrnn', 'model_name': 'VHER', 'model_number': '1501', 'player_info': ''}, ]

    colors = cm.rainbow(np.linspace(0, 1, len(validated_model_type)))

    model_category_msg_all = []
    event_numbers_all = []
    acc_diff_mean_by_event_all = []
    acc_diff_var_by_event_all = []
    game_time_all = []
    acc_diff_mean_by_time_all = []
    acc_diff_var_by_time_all = []
    for model_info_setting in validated_model_type:

        model_category = model_info_setting.get('model_category')
        model_name = model_info_setting.get('model_name')
        model_number = model_info_setting.get('model_number')
        player_info = model_info_setting.get('player_info')
        if model_category == 'cvrnn':
            embed_mode = '_embed_random'
            predicted_target = '_PlayerLocalId_predict_nex_goal'
            icehockey_cvrnn_config_path = "../../environment_settings/icehockey_cvrnn{0}_config{1}{2}.yaml".format(
                predicted_target, player_info, embed_mode)
            icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
        elif model_category == 'cvae':
            predicted_target = '_PlayerLocalId_predict_next_goal'  # playerId_
            # player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
            icehockey_config_path = "../../environment_settings/icehockey_cvae_lstm{0}_config{1}.yaml".format(
                predicted_target, player_info)
            icehockey_model_config = CVAECongfig.load(icehockey_config_path)
        elif model_category == 'vhe':
            predicted_target = '_PlayerLocalId_predict_next_goal'  # playerId_
            # player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
            icehockey_config_path = "../../environment_settings/icehockey_vhe_lstm{0}_config{1}.yaml".format(
                predicted_target, player_info)
            icehockey_model_config = CVAECongfig.load(icehockey_config_path)
        elif model_category == 'encoder':
            predicted_target = '_PlayerLocalId_predict_next_goal'
            # player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
            icehockey_encoder_config_path = "../../environment_settings/" \
                                            "icehockey_stats_lstm_encoder{0}" \
                                            "_config{1}.yaml".format(predicted_target, player_info)
            icehockey_model_config = EncoderConfig.load(icehockey_encoder_config_path)
        elif model_category == 'lstm_diff':
            icehockey_config_path = "../../environment_settings/" \
                                    "ice_hockey_predict_score_diff_lstm{0}.yaml".format(player_info)
            icehockey_model_config = LSTMDiffCongfig.load(icehockey_config_path)
            player_id_cluster_dir = None
        else:
            raise ValueError("uknown model catagoery {0}".format(model_category))

        if model_category != 'zero':
            data_name = get_data_name(icehockey_model_config, model_category, model_number)
            data_name = data_name.replace('Qs', 'accumu_Qs') + "_cv"
        else:
            data_name = None

        print(model_category + '_' + model_number + player_info)
        acc_diff_mean_by_event, acc_diff_var_by_event, event_numbers, \
        acc_diff_mean_by_time, acc_diff_var_by_time, game_time_list = validate_score_diff(model_data_store_dir,
                                                                                          data_name,
                                                                                          model_data_store_dir,
                                                                                          data_store,
                                                                                          model_category,
                                                                                          None,
                                                                                          cv_number=5)
        event_numbers_all.append(event_numbers)
        acc_diff_mean_by_event_all.append(acc_diff_mean_by_event)
        acc_diff_var_by_event_all.append(acc_diff_var_by_event)
        model_category_msg_all.append(model_name)
        game_time_all.append(game_time_list)
        acc_diff_mean_by_time_all.append(acc_diff_mean_by_time)
        acc_diff_var_by_time_all.append(acc_diff_var_by_time)
        # model_category_msg_all.append(model_name + '_' + model_number + player_info)
        # model_category_msg_all.append(model_name + player_info)
        # testing_file.close()

    # plot_diff(game_time_list=event_numbers_all, diff_values_list=acc_diff_mean_all,
    #           model_category_all=model_category_msg_all)

    plot_cv_diff(game_time_list=event_numbers_all,
                 diff_mean_values_list=acc_diff_mean_by_event_all,
                 diff_var_values_list=acc_diff_var_by_event_all,
                 model_category_all=model_category_msg_all,
                 colors=colors)

    # plot_cv_diff(game_time_list=game_time_all,
    #              diff_mean_values_list=acc_diff_mean_by_time_all,
    #              diff_var_values_list=acc_diff_var_by_time_all,
    #              model_category_all=model_category_msg_all,
    #              colors=colors)
