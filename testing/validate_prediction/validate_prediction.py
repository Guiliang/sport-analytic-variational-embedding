import sys

from config.stats_encoder_config import EncoderConfig

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.lstm_prediction_config import LSTMPredictConfig
from support.model_tools import get_model_and_log_name, get_data_name, \
    validate_games_prediction
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    local_test_flag = False
    model_category = 'cvae'
    model_number = 901
    player_info = ''
    embed_mode = ''
    rnn_type = ''

    if model_category == 'cvrnn':
        embed_mode = '_embed_random'
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_cvrnn_config_path = "../../environment_settings/icehockey_cvrnn{0}_config{1}{2}.yaml". \
            format(predicted_target, player_info, embed_mode)
        icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
    elif model_category == 'lstm_prediction':
        icehockey_config_path = "../../environment_settings/ice_hockey_ActionGoal_prediction{0}.yaml".format(
            player_info)
        player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_model_config = LSTMPredictConfig.load(icehockey_config_path)
    elif model_category == 'vhe':
        rnn_type = '_lstm'
        predicted_target = '_PlayerLocalId_predict_next_goal'  # playerId_
        player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_config_path = "../../environment_settings/icehockey_vhe{2}{0}_config{1}.yaml" \
            .format(predicted_target,
                    player_info,
                    rnn_type)
        icehockey_model_config = CVAECongfig.load(icehockey_config_path)
    elif model_category == 'cvae':
        rnn_type = '_lstm'
        predicted_target = '_PlayerLocalId_predict_next_goal'  # playerId_
        player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_config_path = "../../environment_settings/icehockey_cvae{2}{0}_config{1}.yaml" \
            .format(predicted_target,
                    player_info,
                    rnn_type)
        icehockey_model_config = CVAECongfig.load(icehockey_config_path)
        # testing_file = open('./LSTM_diff{1}_model{2}_testing_results{0}.txt'. \
        #                     format(datetime.date.today().strftime("%Y%B%d"), '', str(model_number)), 'wb')
    elif model_category == 'encoder':
        rnn_type = '_lstm'
        predicted_target = '_PlayerLocalId_predict_next_goal'
        player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_encoder_config_path = "../../environment_settings/" \
                                        "icehockey_stats{1}_encoder{0}" \
                                        "_config{2}.yaml".format(predicted_target, rnn_type, player_info)
        icehockey_model_config = EncoderConfig.load(icehockey_encoder_config_path)

    # elif model_category == 'encoder_lstm':
    #     rnn_type = '_lstm'
    #     predicted_target = '_PlayerLocalId_predict_next_goal'
    #     player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
    #     icehockey_encoder_config_path = "../../environment_settings/" \
    #                                     "icehockey_stats{1}_encoder{0}" \
    #                                     "_config.yaml".format(predicted_target, rnn_type, player_info)
    #     icehockey_model_config = EncoderConfig.load(icehockey_encoder_config_path)

    else:
        raise ValueError("uknown model catagoery {0}".format(model_category))

    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                        model_catagoery=model_category)

    testing_dir_games_all = []
    with open('../../sport_resource/ice_hockey_201819/testing_file_dirs_all.csv', 'rb') as f:
        # with open(saved_network_dir + '/testing_file_dirs_all.csv', 'rb') as f:
        testing_dir_all = f.readlines()
    for testing_dir in testing_dir_all:
        testing_dir_games_all.append(str(int(testing_dir)))
    model_data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
    source_data_dir = '/Local-Scratch/oschulte/Galen/2018-2019/'

    # data_name = get_data_name(icehockey_model_config, model_category, model_number)

    print(model_category + '_' + str(model_number) + player_info)

    with open('./results/prediction_acc_' + model_category + rnn_type +
              '_' + str(model_number) + player_info+embed_mode,
              'wb') as file_writer:

        if local_test_flag:
            data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        else:
            data_store_dir = icehockey_model_config.Learn.save_mother_dir \
                             + "/oschulte/Galen/Ice-hockey-data/2018-2019/"

        validate_games_prediction(config=icehockey_model_config,
                                  data_store_dir=data_store_dir,
                                  source_data_dir=source_data_dir,
                                  dir_all=testing_dir_games_all,
                                  # model_nn=model_nn,
                                  # sess_nn=sess_nn,
                                  # model_path=model_path,
                                  player_basic_info_dir='../../sport_resource/ice_hockey_201819/player_info_2018_2019.json',
                                  game_date_dir='../../sport_resource/ice_hockey_201819/game_dates_2018_2019.json',
                                  player_box_score_dir='../../sport_resource/ice_hockey_201819/Scale_NHL_players_game_summary_201819.csv',
                                  model_number=model_number,
                                  player_id_cluster_dir=player_id_cluster_dir,
                                  saved_network_dir=saved_network_dir,
                                  model_category=model_category,
                                  file_writer=file_writer)
