import os
from random import shuffle

from config.cvrnn_config import CVRNNCongfig
from support.model_tools import compute_values_for_all_games

if __name__ == '__main__':
    test_flag = False
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

    icehockey_cvrnn_config_path = "../environment_settings/icehockey_cvrnn{0}_config.yaml".format(predicted_target)
    icehockey_cvrnn_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)

    if test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = icehockey_cvrnn_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        shuffle(dir_games_all)  # randomly shuffle the list

    compute_values_for_all_games(config=icehockey_cvrnn_config,
                                 data_store_dir=data_store_dir,
                                 dir_all=dir_games_all,
                                 model_number=901,
                                 player_id_cluster_dir=player_id_cluster_dir,
                                 model_category='cvrnn')