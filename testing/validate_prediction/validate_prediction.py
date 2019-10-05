import sys
print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.lstm_prediction_config import LSTMPredictConfig
from support.model_tools import get_model_and_log_name, get_data_name, \
    validate_games_prediction

if __name__ == '__main__':
    local_test_flag = False
    model_category = 'lstm_prediction'
    model_number = 1501
    player_info = '_box'

    if model_category == 'cvrnn':
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        player_id_cluster_dir = '../../resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_cvrnn_config_path = "../../environment_settings/icehockey_cvrnn{0}_config{1}.yaml". \
            format(predicted_target, player_info)
        icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
    elif model_category == 'lstm_prediction':
        icehockey_config_path = "../../environment_settings/ice_hockey_ActionGoal_prediction{0}.yaml".format(player_info)
        player_id_cluster_dir = '../../resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_model_config = LSTMPredictConfig.load(icehockey_config_path)
    elif model_category == 'cvae':
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        player_id_cluster_dir = '../../resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_config_path = "../../environment_settings/icehockey_cvae{0}_config.yaml".format(
            predicted_target)
        icehockey_model_config = CVAECongfig.load(icehockey_config_path)
        # testing_file = open('./LSTM_diff{1}_model{2}_testing_results{0}.txt'. \
        #                     format(datetime.date.today().strftime("%Y%B%d"), '', str(model_number)), 'wb')
    else:
        raise ValueError("uknown model catagoery {0}".format(model_category))

    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                        model_catagoery=model_category)

    testing_dir_games_all = []
    # with open('../../resource/ice_hockey_201819/testing_file_dirs_all.csv', 'rb') as f:
    with open(saved_network_dir + '/testing_file_dirs_all.csv', 'rb') as f:
        testing_dir_all = f.readlines()
    for testing_dir in testing_dir_all:
        testing_dir_games_all.append(str(int(testing_dir)))
    model_data_store_dir = icehockey_model_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
    source_data_dir = '/Local-Scratch/oschulte/Galen/2018-2019/'

    # data_name = get_data_name(icehockey_model_config, model_category, model_number)

    print(model_category + '_' + str(model_number) + player_info)

    with open('prediction_acc_' + model_category + '_' + str(model_number) + player_info, 'wb') as file_writer:

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
                                  player_basic_info_dir='../../resource/ice_hockey_201819/player_info_2018_2019.json',
                                  game_date_dir='../../resource/ice_hockey_201819/game_dates_2018_2019.json',
                                  player_box_score_dir='../../resource/ice_hockey_201819/Scale_NHL_players_game_summary_201819.csv',
                                  model_number=model_number,
                                  player_id_cluster_dir=player_id_cluster_dir,
                                  saved_network_dir=saved_network_dir,
                                  model_category=model_category,
                                  file_writer=file_writer)
