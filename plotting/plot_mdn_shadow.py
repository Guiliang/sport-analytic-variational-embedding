import os
import tensorflow as tf
import numpy as np
from config.mdn_Qs_config import MDNQsCongfig
from nn_structure.mdn_nn import MixtureDensityNN
from support.data_processing_tools import get_icehockey_game_data, transfer2seq
from support.model_tools import get_model_and_log_name, load_nn_model
from support.plot_tools import plot_shadow


def gather_plot_values(dir_games_all, data_store, config, model):

    mu_all = []
    var_all = []

    for dir_game in dir_games_all:
        if dir_game == '.DS_Store':
            continue
        state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
            data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=None)
        action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                  max_length=config.Learn.max_seq_length)
        team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                   max_length=config.Learn.max_seq_length)
        player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                        max_length=config.Learn.max_seq_length)

        input_data = np.concatenate([state_input, action_seq], axis=2)

        [mu, var] = sess.run([model.mu_out, model.var_out],
                             feed_dict={model.rnn_input_ph: input_data,
                                        model.trace_lengths_ph: state_trace_length})
        mu_all.append(mu)
        var_all.append(var)
    return mu_all, var_all


if __name__ == '__main__':
    test_flag = True
    ci_value = 1.96
    icehockey_mdn_Qs_config_path = "../environment_settings/ice_hockey_predict_Qs_mdn.yaml"
    icehockey_mdn_Qs_config = MDNQsCongfig.load(icehockey_mdn_Qs_config_path)
    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_mdn_Qs_config, model_catagoery='mdn_Qs')

    if test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
        save_flag = False
    else:
        data_store_dir = icehockey_mdn_Qs_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 9]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: testing
        save_flag = True
    number_of_total_game = len(dir_games_all)
    icehockey_mdn_Qs_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    model = MixtureDensityNN(config=icehockey_mdn_Qs_config)
    model()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    load_nn_model(saver, sess, saved_network_dir)

    mu_all, var_all = gather_plot_values(dir_games_all, data_store_dir, icehockey_mdn_Qs_config, model)

    plot_shadow(range(len(mu_all[0])), mu_all[0],
                mu_all[0]-ci_value*var_all[0], mu_all[0]+ci_value*var_all[0])
