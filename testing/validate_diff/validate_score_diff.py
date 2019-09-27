import datetime
import json
import numpy as np

from config.LSTM_diff_config import LSTMDiffCongfig
from config.cvrnn_config import CVRNNCongfig
from support.data_processing_tools import read_feature_within_events
from support.model_tools import get_model_and_log_name, get_data_name
from support.plot_tools import plot_diff


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


def validate_score_diff(model_data_store_dir,
                        testing_dir_games_all,
                        data_name,
                        source_data_dir,
                        data_store,
                        file_writer=None):
    real_label_record = np.ones([len(testing_dir_games_all), 5000]) * -1
    output_label_record = np.ones([len(testing_dir_games_all), 5000]) * -1

    acc_diff = []
    event_numbers = []

    for dir_index in range(0, len(testing_dir_games_all)):
        print('Processing game {0}'.format(dir_index))
        testing_dir = testing_dir_games_all[dir_index]
        model_values = obtain_model_predictions(model_data_store_dir, testing_dir, data_name)

        score_difference_game = read_feature_within_events(testing_dir,
                                                           data_store,
                                                           'scoreDifferential',
                                                           transfer_home_number=True,
                                                           data_store=source_data_dir)
        real_label_all = [score_difference_game[-1]] * len(model_values)
        output_label_all = model_values[:, 0] - model_values[:, 1] + score_difference_game[:len(model_values)]

        real_label_record[dir_index][:len(real_label_all)] = real_label_all[:len(real_label_all)]
        output_label_record[dir_index][:len(output_label_all)] = output_label_all[:len(output_label_all)]

    for i in range(0, output_label_record.shape[1]):
        real_outcome_record_step = real_label_record[:, i]
        model_output_record_step = output_label_record[:, i]
        diff_sum = 0
        total_number = 0
        print_flag = True
        for win_index in range(0, len(real_outcome_record_step)):
            if model_output_record_step[win_index] == -100 or real_outcome_record_step[win_index] == -100:
                print_flag = True
                continue
            diff = abs(model_output_record_step[win_index] - real_outcome_record_step[win_index])
            diff_sum += diff
            total_number += 1
        acc_diff.append(float(diff_sum) / total_number)
        event_numbers.append(i)

        if file_writer is not None:
            file_writer.write('diff of time {0} is {1}\n'.format(str(i), str(float(diff_sum) / total_number)))

        if print_flag:
            if i % 100 == 0 and total_number > 0:
                print('diff of time {0} is {1}'.format(str(i), str(float(diff_sum) / total_number)))
    return acc_diff, event_numbers


if __name__ == '__main__':
    # cvrnn_result_dir = '../interface/cvrnn_testing_results2019September23.txt'
    # game_time, cvrnn_results = read_results_values(cvrnn_result_dir)
    # plot_diff(game_time, cvrnn_results)
    model_number = '2101'
    model_category_all = ['cvrnn', 'lstm_diff']
    event_numbers_all = []
    acc_diff_all = []
    for model_category in model_category_all:
        if model_category == 'cvrnn':
            box_msg = '_box'
            predicted_target = '_PlayerLocalId'  # playerId_
            icehockey_cvrnn_config_path = "../environment_settings/icehockey_cvrnn{0}_config{1}.yaml". \
                format(predicted_target, box_msg)
            icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
            testing_file = open('./cvrnn{1}_model{2}_testing_results{0}.txt'. \
                                format(datetime.date.today().strftime("%Y%B%d"), box_msg, str(model_number)), 'wb')
        elif model_category == 'lstm_diff':
            icehockey_config_path = "../environment_settings/ice_hockey_predict_score_diff_lstm.yaml"
            icehockey_model_config = LSTMDiffCongfig.load(icehockey_config_path)
            testing_file = open('./LSTM_diff{1}_model{2}_testing_results{0}.txt'. \
                                format(datetime.date.today().strftime("%Y%B%d"), '', str(model_number)), 'wb')
        else:
            raise ValueError("uknown model catagoery {0}".format(model_category))

        saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config, model_catagoery=model_category)
        testing_dir_games_all = []
        with open(saved_network_dir + '/testing_file_dirs_all.csv', 'rb') as f:
            testing_dir_all = f.readlines()
        for testing_dir in testing_dir_all:
            testing_dir_games_all.append(str(int(testing_dir)))
        model_data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        data_store = '/Local-Scratch/oschulte/Galen/2018-2019/'

        data_name = get_data_name(icehockey_model_config, model_category, model_number)

        acc_diff, event_numbers = validate_score_diff(model_data_store_dir,
                                                      testing_dir_games_all,
                                                      data_name.replace('Qs', 'accumu_Qs'),
                                                      model_data_store_dir,
                                                      data_store,
                                                      testing_file)
        event_numbers_all.append(event_numbers)
        acc_diff_all.append(acc_diff)
        testing_file.close()

    plot_diff(game_time_list=event_numbers_all, diff_values_list=acc_diff_all, model_category_all=model_category_all)
