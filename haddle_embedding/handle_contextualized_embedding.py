import json
import random
import os
from scipy import stats
import tensorflow as tf
from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.lstm_prediction_config import LSTMPredictConfig
from config.stats_encoder_config import EncoderConfig
from support.embedding_tools import plot_embeddings, dimensional_reduction, aggregate_positions_within_cluster, \
    get_player_cluster
from support.model_tools import get_model_and_log_name, get_data_name, validate_games_player_id, \
    validate_games_embedding


def illustrate_embeddings(encoder_values, player_index_list, player_basic_info_dir):
    position_list = get_player_cluster(player_index_list=player_index_list,
                                       player_basic_info_dir=player_basic_info_dir, clutser_type='pindex')
    dr_embedding = dimensional_reduction(encoder_values, dr_method='TSNE')
    plot_embeddings(data=dr_embedding, cluster_number=position_list)

    pass


def significance_test(testing_target, testing_objects):
    # testing_target = testing_target[:, :128]
    for testing_object in testing_objects:
        a = testing_target.flatten()
        b = testing_object.flatten()
        results = stats.ttest_rel(testing_target[:, :128].flatten(), testing_object[:, :128].flatten())
        print(results)
    return


if __name__ == '__main__':
    local_test_flag = False
    # model_type = 'cvrnn'
    model_type_all = ['cvrnn', 'cvae', 'vhe', 'encoder']
    model_number = 1801
    player_info = ''
    embed_mode = '_'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    player_basic_info_dir = '../sport_resource/ice_hockey_201819/player_info_2018_2019.json'
    game_date_dir = '../sport_resource/ice_hockey_201819/game_dates_2018_2019.json'
    player_box_score_dir = '../sport_resource/ice_hockey_201819/Scale_NHL_players_game_summary_201819.csv'
    data_store_dir = "/Local-Scratch/oschulte/Galen/Ice-hockey-data/2018-2019/"
    dir_games_all = os.listdir(data_store_dir)
    selected_game_index = random.sample(range(0, len(dir_games_all)), 100)
    testing_dir_games_all = [dir_games_all[index] for index in selected_game_index]

    all_type_embedding = []

    for model_type in model_type_all:

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

        elif model_type == 'encoder':
            predicted_target = '_PlayerLocalId_predict_next_goal'
            player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
            icehockey_encoder_config_path = "../environment_settings/" \
                                            "icehockey_stats_lstm_encoder{0}" \
                                            "_config{1}.yaml".format(predicted_target, player_info)
            icehockey_model_config = EncoderConfig.load(icehockey_encoder_config_path)
        else:
            raise ValueError("uknown model catagoery {0}".format(model_type))

        saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                            model_catagoery=model_type,
                                                            running_number=0)

        # data_name = get_data_name(icehockey_model_config, model_category, model_number)

        print(model_type + '_' + str(model_number) + player_info)

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
                                                    model_category=model_type,
                                                    sess_nn=None)
        all_type_embedding.append(all_embedding)

    significance_test(testing_target=all_type_embedding[0], testing_objects=all_type_embedding[1:])

    # illustrate_embeddings(encoder_values=all_embedding,
    #                       player_index_list=all_player_index,
    #                       player_basic_info_dir=player_basic_info_dir)
