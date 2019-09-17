import os
from random import shuffle

from config.LSTM_Qs_config import LSTMQsCongfig
from config.cvrnn_config import CVRNNCongfig
from support.model_tools import compute_games_Q_values

if __name__ == '__main__':

    model_type = 'cvrnn'
    test_flag = False
    if model_type == 'cvrnn':
        player_id_type = 'local_id'
        if player_id_type == 'ap_cluster':
            player_id_cluster_dir = '../resource/ice_hockey_201819/player_id_ap_cluster.json'
            predicted_target = '_PlayerPositionClusterAP'  # playerId_
        elif player_id_type == 'km_cluster':
            player_id_cluster_dir = '../resource/ice_hockey_201819/player_id_km_cluster.json'
            predicted_target = '_PlayerPositionClusterKM'  # playerId_
        elif player_id_type == 'local_id':
            player_id_cluster_dir = '../resource/ice_hockey_201819/local_player_id_2018_2019.json'
            predicted_target = '_PlayerLocalId'  # playerId_
        else:
            player_id_cluster_dir = None
            predicted_target = ''

        icehockey_config_path = "../environment_settings/icehockey_cvrnn{0}_config.yaml".format(predicted_target)
        icehockey_model_config = CVRNNCongfig.load(icehockey_config_path)
        model_number = 901

        icehockey_model_config.Learn.apply_box_score = False  # TODO: tmp setting, remove me
        icehockey_model_config.Arch.CVRNN.y_dim = 70  # TODO: tmp setting, remove me

    elif model_type == 'lstm_Qs':
        icehockey_config_path = "../environment_settings/ice_hockey_predict_Qs_lstm.yaml"
        icehockey_model_config = LSTMQsCongfig.load(icehockey_config_path)
        player_id_cluster_dir = None
        model_number = 901
    else:
        raise ValueError('incorrect model type {0}'.format(model_type))

    if test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        shuffle(dir_games_all)  # randomly shuffle the list

    compute_games_Q_values(config=icehockey_model_config,
                           data_store_dir=data_store_dir,
                           dir_all=dir_games_all,
                           model_number=model_number,
                           player_id_cluster_dir=player_id_cluster_dir,
                           model_category=model_type)