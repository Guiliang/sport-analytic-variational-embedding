import json
import os
import tensorflow as tf
from random import shuffle

from config.LSTM_Qs_config import LSTMQsCongfig
from config.LSTM_diff_config import LSTMDiffCongfig
from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.stats_encoder_config import EncoderConfig
from support.model_tools import compute_games_Q_values, get_model_and_log_name, validate_model_initialization, \
    get_data_name

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    model_type = 'cvrnn'
    player_info = ''
    model_number = 2101
    local_test_flag = False
    if model_type == 'cvrnn':
        embed_mode = '_embed_random'
        predicted_target = '_PlayerLocalId_predict_nex_goal'
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_cvrnn_config_path = "../environment_settings/icehockey_cvrnn{0}_config{1}{2}.yaml".format(
            predicted_target, player_info, embed_mode)
        icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)

    elif model_type == 'cvae':
        predicted_target = '_PlayerLocalId_predict_next_goal'  # playerId_
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_config_path = "../environment_settings/icehockey_cvae_lstm{0}_config{1}.yaml".format(
            predicted_target, player_info)
        icehockey_model_config = CVAECongfig.load(icehockey_config_path)

    elif model_type == 'vhe':
        predicted_target = '_PlayerLocalId_predict_next_goal'  # playerId_
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_config_path = "../environment_settings/icehockey_vhe_lstm{0}_config{1}.yaml".format(
            predicted_target, player_info)
        icehockey_model_config = CVAECongfig.load(icehockey_config_path)

    elif model_type == 'lstm_Qs':
        icehockey_config_path = "../environment_settings/ice_hockey_predict_Qs_lstm{0}.yaml".format(player_info)
        icehockey_model_config = LSTMQsCongfig.load(icehockey_config_path)
        player_id_cluster_dir = None

    elif model_type == 'encoder':
        predicted_target = '_PlayerLocalId_predict_next_goal'
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_encoder_config_path = "../environment_settings/" \
                                        "icehockey_stats_lstm_encoder{0}" \
                                        "_config{1}.yaml".format(predicted_target, player_info)
        icehockey_model_config = EncoderConfig.load(icehockey_encoder_config_path)

    elif model_type == 'lstm_diff':
        icehockey_config_path = "../environment_settings/ice_hockey_predict_score_diff_lstm{0}.yaml".format(player_info)
        icehockey_model_config = LSTMDiffCongfig.load(icehockey_config_path)
        player_id_cluster_dir = None
    else:
        raise ValueError('incorrect model type {0}'.format(model_type))

    if local_test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)

    sess_nn = tf.InteractiveSession()
    model_nn = validate_model_initialization(sess_nn=sess_nn, model_category=model_type,
                                             config=icehockey_model_config)

    running_numbers = [0, 1, 2, 3, 4]

    # cv_record_all_model_next_Q_values = []
    # cv_record_all_model_accumu_Q_values = []
    #
    # for dir_game_index in range(0, len(dir_games_all)):
    #     game_cv_record = {}
    #     for running_number in running_numbers:
    #         game_cv_record.update({running_number: None})
    #     cv_record_all_model_next_Q_values.append({dir_game_index: game_cv_record})
    #     cv_record_all_model_accumu_Q_values.append({dir_game_index: game_cv_record})

    for running_number in running_numbers:
        saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                            model_catagoery=model_type,
                                                            running_number=running_number)

        model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)

        compute_games_Q_values(config=icehockey_model_config,
                               data_store_dir=data_store_dir,
                               dir_all=dir_games_all,
                               model_nn=model_nn,
                               sess_nn=sess_nn,
                               model_path=model_path,
                               model_number=model_number,
                               player_id_cluster_dir=player_id_cluster_dir,
                               model_category=model_type,
                               return_values_flag=False,
                               apply_cv=True,
                               running_number=running_number)
        # for dir_game_index in range(0, len(dir_games_all)):
        #     cv_record_all_model_next_Q_values[dir_game_index].update(
        #         {running_number: model_next_Q_values_all[dir_game_index]})
        #     cv_record_all_model_accumu_Q_values[dir_game_index].update(
        #         {running_number: model_accumu_Q_value_all[dir_game_index]})

    for dir_game_index in range(0, len(dir_games_all)):
        data_name = get_data_name(config=icehockey_model_config,
                                  model_catagoery=model_type,
                                  model_number=model_number)
        game_name_dir = dir_games_all[dir_game_index]
        game_store_dir = game_name_dir.split('.')[0]
        game_all_next_Qs_values = {}
        game_all_accumu_Qs_values = {}
        for running_number in running_numbers:
            with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'next_Qs')
                      + '_r'+str(running_number), 'r') as outfile:
                cv_next_Qs_game_values = json.load(outfile)
            game_all_next_Qs_values.update({running_number: cv_next_Qs_game_values})
            os.remove(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'next_Qs')
                      + '_r'+str(running_number))

            with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'accumu_Qs')
                      + '_r'+str(running_number), 'r') as outfile:
                cv_accumu_Qs_game_values = json.load(outfile)
            game_all_accumu_Qs_values.update({running_number: cv_accumu_Qs_game_values})
            os.remove(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'accumu_Qs')
                      + '_r'+str(running_number))

        with open(data_store_dir + "/" + game_store_dir + "/"
                  + data_name.replace('Qs', 'next_Qs')+'_cv', 'w') as outfile:
            json.dump(game_all_next_Qs_values, outfile)
        with open(data_store_dir + "/" + game_store_dir + "/"
                  + data_name.replace('Qs', 'accumu_Qs')+'_cv', 'w') as outfile:
            json.dump(game_all_accumu_Qs_values, outfile)
