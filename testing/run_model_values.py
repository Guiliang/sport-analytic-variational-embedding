import os
from random import shuffle

from config.LSTM_Qs_config import LSTMQsCongfig
from config.LSTM_diff_config import LSTMDiffCongfig
from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.stats_encoder_config import EncoderConfig
from support.model_tools import compute_games_Q_values

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    model_type = 'encoder'
    player_info = '_box'
    model_number = 1801
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

    compute_games_Q_values(config=icehockey_model_config,
                           data_store_dir=data_store_dir,
                           dir_all=dir_games_all,
                           model_number=model_number,
                           player_id_cluster_dir=player_id_cluster_dir,
                           model_category=model_type)
