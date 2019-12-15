import json
import os

from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.lstm_prediction_config import LSTMPredictConfig
from config.stats_encoder_config import EncoderConfig
from support.embedding_tools import plot_embeddings, dimensional_reduction, aggregate_positions_within_cluster, \
    get_player_cluster
from support.model_tools import get_model_and_log_name, get_data_name, validate_games_player_id, \
    validate_games_embedding


def handle_embeddings(encoder_values, player_index_list, player_basic_info_dir):
    position_list = get_player_cluster(player_index_list=player_index_list,
                                       player_basic_info_dir=player_basic_info_dir, clutser_type='pindex')
    dr_embedding = dimensional_reduction(encoder_values, dr_method='TSNE')
    plot_embeddings(data=dr_embedding, cluster_number=position_list)

    pass


if __name__ == '__main__':
    local_test_flag = False
    model_category = 'cvrnn'
    model_number = 2101
    player_info = ''
    embed_mode = '_'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    player_basic_info_dir = '../sport_resource/ice_hockey_201819/player_info_2018_2019.json'
    game_date_dir = '../sport_resource/ice_hockey_201819/game_dates_2018_2019.json'
    player_box_score_dir = '../sport_resource/ice_hockey_201819/Scale_NHL_players_game_summary_201819.csv'

    if model_category == 'cvrnn':
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_cvrnn_config_path = "../environment_settings/icehockey_cvrnn{0}_config{1}{2}.yaml". \
            format(predicted_target, player_info, embed_mode)
        icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
    elif model_category == 'cvae':
        predicted_target = '_PlayerLocalId_predict_next_goal'  # playerId_
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_config_path = "../environment_settings/icehockey_cvae{0}_config{1}.yaml".format(
            predicted_target, player_info)
        icehockey_model_config = CVAECongfig.load(icehockey_config_path)
    elif model_category == 'encoder':
        rnn_type = ''
        predicted_target = '_PlayerLocalId_predict_next_goal'
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_encoder_config_path = "../environment_settings/" \
                                        "icehockey_stats{1}_encoder{0}" \
                                        "_config{2}.yaml".format(predicted_target, rnn_type, player_info)
        icehockey_model_config = EncoderConfig.load(icehockey_encoder_config_path)
    elif model_category == 'lstm_prediction':
        predicted_target = '_PlayerLocalId'
        icehockey_config_path = "../environment_settings/" \
                                "ice_hockey_PlayerLocalId_prediction{0}.yaml".format(player_info)
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_model_config = LSTMPredictConfig.load(icehockey_config_path)
    else:
        raise ValueError("uknown model catagoery {0}".format(model_category))

    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                        model_catagoery=model_category)

    testing_dir_games_all = []
    # with open('../../sport_resource/ice_hockey_201819/testing_file_dirs_all.csv', 'rb') as f:
    with open(saved_network_dir + '/testing_file_dirs_all.csv', 'rb') as f:
        testing_dir_all = f.readlines()
    for testing_dir in testing_dir_all:
        testing_dir_games_all.append(str(int(testing_dir)))
    model_data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
    data_store = '/Local-Scratch/oschulte/Galen/2018-2019/'

    # data_name = get_data_name(icehockey_model_config, model_category, model_number)

    print(model_category + '_' + str(model_number) + player_info)

    if local_test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
    else:
        data_store_dir = icehockey_model_config.Learn.save_mother_dir \
                         + "/oschulte/Galen/Ice-hockey-data/2018-2019/"

    all_embedding, \
    all_player_index = validate_games_embedding(config=icehockey_model_config,
                                                data_store_dir=data_store_dir,
                                                dir_all=testing_dir_games_all[:1],
                                                player_basic_info_dir=player_basic_info_dir,
                                                game_date_dir=game_date_dir,
                                                player_box_score_dir=player_box_score_dir,
                                                model_number=model_number,
                                                player_id_cluster_dir=player_id_cluster_dir,
                                                saved_network_dir=saved_network_dir,
                                                model_category=model_category)

    handle_embeddings(encoder_values=all_embedding[0],
                      player_index_list=all_player_index[0],
                      player_basic_info_dir=player_basic_info_dir)
