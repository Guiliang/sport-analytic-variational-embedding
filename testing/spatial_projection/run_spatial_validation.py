from config.cvae_config import CVAECongfig
from config.cvrnn_config import CVRNNCongfig
from config.lstm_prediction_config import LSTMPredictConfig
from support.data_processing_tools import start_lstm_generate_spatial_simulation
from support.model_tools import validate_games_prediction, get_model_and_log_name
from support.plot_tools import plot_spatial_projection


def generate_spatial_simulation_data():
    simulation_type = 'spatial'
    history_action_type = []
    history_action_type_coord = []
    action_type = 'shot'
    data_simulation_dir = './simulation_data_all/'

    simulated_data_dir_all = start_lstm_generate_spatial_simulation(history_action_type, history_action_type_coord,
                                                                    action_type, data_simulation_dir, simulation_type,
                                                                    feature_type='v1', max_trace_length=10,
                                                                    is_home=True)

    return simulated_data_dir_all


def compute_next_goal_model_values(config, data_store_dir, source_data_dir,
                                   model_category, saved_network_dir, model_number):
    all_prediction_output = validate_games_prediction(config,
                                                      data_store_dir,
                                                      source_data_dir,
                                                      model_category=model_category,
                                                      prediction_type='spatial_simulation',
                                                      saved_network_dir=saved_network_dir,
                                                      model_number=model_number)

    plot_spatial_projection(value_spatial=all_prediction_output[:, :, 0])


if __name__ == '__main__':

    local_test_flag = False
    model_category = 'lstm_prediction'
    model_number = 1801
    player_info = ''

    if model_category == 'cvrnn':
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        # player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_cvrnn_config_path = "../../environment_settings/icehockey_cvrnn{0}_config{1}.yaml". \
            format(predicted_target, player_info)
        icehockey_model_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)
    elif model_category == 'lstm_prediction':
        icehockey_config_path = "../../environment_settings/ice_hockey_ActionGoal_prediction{0}.yaml".format(
            player_info)
        # player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_model_config = LSTMPredictConfig.load(icehockey_config_path)
    elif model_category == 'cvae':
        predicted_target = '_PlayerLocalId_predict_nex_goal'  # playerId_
        # player_id_cluster_dir = '../../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        icehockey_config_path = "../../environment_settings/icehockey_cvae{0}_config.yaml".format(
            predicted_target)
        icehockey_model_config = CVAECongfig.load(icehockey_config_path)
        # testing_file = open('./LSTM_diff{1}_model{2}_testing_results{0}.txt'. \
        #                     format(datetime.date.today().strftime("%Y%B%d"), '', str(model_number)), 'wb')
    else:
        raise ValueError("uknown model catagoery {0}".format(model_category))

    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_model_config,
                                                        model_catagoery=model_category)

    # data_store_dir_all = generate_spatial_simulation_data()
    data_store_dir = './simulation_data_all/spatial/LSTM_Home_spatial-shot-[]-featurev1'

    if local_test_flag:
        source_data_dir = ''
    else:
        source_data_dir = '/Local-Scratch/oschulte/Galen/2018-2019/'

    # for data_store_dir in data_store_dir_all:
    compute_next_goal_model_values(config=icehockey_model_config,
                                   data_store_dir=data_store_dir,
                                   source_data_dir=source_data_dir,
                                   model_category=model_category,
                                   saved_network_dir=saved_network_dir,
                                   model_number=model_number)
