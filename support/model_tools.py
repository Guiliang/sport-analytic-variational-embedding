import random

import tensorflow as tf
import numpy as np
import json

from nn_structure.cvrnn import CVRNN
from nn_structure.lstm_Qs_nn import TD_Prediction
from nn_structure.lstm_score_diff_nn import Diff_Prediction
from support.data_processing_tools import get_icehockey_game_data, generate_selection_matrix, transfer2seq


# from support.plot_tools import plot_game_Q_values


class ExperienceReplayBuffer:
    def __init__(self, capacity_number):
        self.capacity = capacity_number
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[random.randint(0, len(self.memory) - 1)]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def load_nn_model(saver, sess, saved_network_dir):
    # saver = tf.train.Saver()
    # merge = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    # sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(saved_network_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # check_point_game_number = int((checkpoint.model_checkpoint_path.split("-"))[-1])
        # game_number_checkpoint = check_point_game_number % config.number_of_total_game
        # game_number = check_point_game_number
        # game_starting_point = 0
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find the network: {0}", format(saved_network_dir))


def get_data_name(config, model_catagoery, model_number):
    player_info = ''
    if config.Learn.apply_box_score:
        player_info += '_box'
    if config.Learn.apply_pid:
        player_info += '_pid'
    if model_catagoery == 'cvrnn':
        data_name = "model_{1}_three_cut_cvrnn_Qs_feature{2}_latent{8}_x{9}_y{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}{12}".format(config.Learn.save_mother_dir,
                                                                                  model_number,
                                                                                  str(config.Learn.feature_type),
                                                                                  str(config.Learn.batch_size),
                                                                                  str(config.Learn.iterate_num),
                                                                                  str(config.Learn.learning_rate),
                                                                                  str(config.Learn.model_type),
                                                                                  str(config.Learn.max_seq_length),
                                                                                  str(config.Arch.CVRNN.latent_dim),
                                                                                  str(config.Arch.CVRNN.y_dim),
                                                                                  str(config.Arch.CVRNN.x_dim),
                                                                                  str(config.Arch.CVRNN.hidden_dim),
                                                                                  player_info
                                                                                  )
    elif model_catagoery == 'lstm_Qs':
        data_name = "model_{1}_three_cut_lstm_Qs_feature{2}_{8}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                    "_dense{11}{12}".format(config.Learn.save_mother_dir,
                                            model_number,
                                            str(config.Learn.feature_type),
                                            str(config.Learn.batch_size),
                                            str(config.Learn.iterate_num),
                                            str(config.Learn.learning_rate),
                                            str(config.Learn.model_type),
                                            str(
                                                config.Learn.max_seq_length),
                                            config.Learn.predict_target,
                                            None,
                                            str(config.Arch.LSTM.h_size),
                                            str(
                                                config.Arch.Dense.hidden_size),
                                            player_info
                                            )
    elif model_catagoery == 'lstm_diff':
        data_name = "model_{1}_three_cut_lstm_Qs_feature{2}_{8}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                    "_dense{11}{12}".format(config.Learn.save_mother_dir,
                                            model_number,
                                            str(config.Learn.feature_type),
                                            str(config.Learn.batch_size),
                                            str(config.Learn.iterate_num),
                                            str(config.Learn.learning_rate),
                                            str(config.Learn.model_type),
                                            str(
                                                config.Learn.max_seq_length),
                                            config.Learn.predict_target,
                                            None,
                                            str(config.Arch.LSTM.h_size),
                                            str(
                                                config.Arch.Dense.hidden_size),
                                            player_info
                                            )

    return data_name


def get_model_and_log_name(config, model_catagoery, train_flag=False, embedding_tag=None):
    if train_flag:
        train_msg = 'Train_'
    else:
        train_msg = ''

    box_info = ''
    if config.Learn.apply_box_score:
        box_info = '_box'

    if model_catagoery == 'cvrnn':  # TODO: add more parameters
        log_dir = "{0}/oschulte/Galen/icehockey-models/cvrnn_log_NN" \
                  "/{1}cvrnn_log_feature{2}_latent{8}_x{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}{12}".format(config.Learn.save_mother_dir,
                                                                                train_msg,
                                                                                str(config.Learn.feature_type),
                                                                                str(config.Learn.batch_size),
                                                                                str(config.Learn.iterate_num),
                                                                                str(config.Learn.learning_rate),
                                                                                str(config.Learn.model_type),
                                                                                str(config.Learn.max_seq_length),
                                                                                str(config.Arch.CVRNN.latent_dim),
                                                                                str(config.Arch.CVRNN.y_dim),
                                                                                # TODO: reorder x_dim and y_dim
                                                                                str(config.Arch.CVRNN.x_dim),
                                                                                str(config.Arch.CVRNN.hidden_dim),
                                                                                box_info
                                                                                )

        saved_network = "{0}/oschulte/Galen/icehockey-models/cvrnn_saved_NN/" \
                        "{1}cvrnn_saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}{12}".format(config.Learn.save_mother_dir,
                                                                                      train_msg,
                                                                                      str(config.Learn.feature_type),
                                                                                      str(config.Learn.batch_size),
                                                                                      str(config.Learn.iterate_num),
                                                                                      str(config.Learn.learning_rate),
                                                                                      str(config.Learn.model_type),
                                                                                      str(config.Learn.max_seq_length),
                                                                                      str(config.Arch.CVRNN.latent_dim),
                                                                                      str(config.Arch.CVRNN.y_dim),
                                                                                      # TODO: reorder x_dim and y_dim
                                                                                      str(config.Arch.CVRNN.x_dim),
                                                                                      str(config.Arch.CVRNN.hidden_dim),
                                                                                      box_info
                                                                                      )
    elif model_catagoery == 'de_embed':
        if embedding_tag is not None:
            train_msg += 'validate{0}_'.format(str(embedding_tag))

        log_dir = "{0}/oschulte/Galen/icehockey-models/de_log_NN" \
                  "/{1}de_embed_log_feature{2}_{8}_embed{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}".format(config.Learn.save_mother_dir,
                                          train_msg,
                                          str(config.Learn.feature_type),
                                          str(config.Learn.batch_size),
                                          str(config.Learn.iterate_num),
                                          str(config.Learn.learning_rate),
                                          str(config.Learn.model_type),
                                          str(config.Learn.max_seq_length),
                                          config.Learn.predict_target,
                                          str(config.Arch.Encode.latent_size),
                                          str(config.Arch.LSTM.h_size),
                                          str(config.Arch.Dense.hidden_node_size),
                                          box_info
                                          )

        saved_network = "{0}/oschulte/Galen/icehockey-models/de_model_saved_NN/" \
                        "{1}de_embed_saved_networks_feature{2}_{8}_embed{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}".format(config.Learn.save_mother_dir,
                                                train_msg,
                                                str(config.Learn.feature_type),
                                                str(config.Learn.batch_size),
                                                str(config.Learn.iterate_num),
                                                str(config.Learn.learning_rate),
                                                str(config.Learn.model_type),
                                                str(config.Learn.max_seq_length),
                                                config.Learn.predict_target,
                                                str(config.Arch.Encode.latent_size),
                                                str(config.Arch.LSTM.h_size),
                                                str(config.Arch.Dense.hidden_node_size),
                                                box_info
                                                )
    elif model_catagoery == 'mdn_Qs':
        log_dir = "{0}/oschulte/Galen/icehockey-models/mdn_Qs_log_NN" \
                  "/{1}mdn_log_feature{2}_{8}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}".format(config.Learn.save_mother_dir,
                                          train_msg,
                                          str(config.Learn.feature_type),
                                          str(config.Learn.batch_size),
                                          str(config.Learn.iterate_num),
                                          str(config.Learn.learning_rate),
                                          str(config.Learn.model_type),
                                          str(config.Learn.max_seq_length),
                                          config.Learn.predict_target,
                                          None,
                                          str(config.Arch.LSTM.h_size),
                                          str(config.Arch.Dense.hidden_size),
                                          box_info
                                          )

        saved_network = "{0}/oschulte/Galen/icehockey-models/mdn_Qs_model_saved_NN/" \
                        "{1}mdn_embed_saved_networks_feature{2}_{8}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            box_info
        )

    elif model_catagoery == 'lstm_Qs':
        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''
        log_dir = "{0}/oschulte/Galen/icehockey-models/lstm_Qs_log_NN" \
                  "/{1}lstm_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}".format(config.Learn.save_mother_dir,
                                              train_msg,
                                              str(config.Learn.feature_type),
                                              str(config.Learn.batch_size),
                                              str(config.Learn.iterate_num),
                                              str(config.Learn.learning_rate),
                                              str(config.Learn.model_type),
                                              str(config.Learn.max_seq_length),
                                              config.Learn.predict_target,
                                              None,
                                              str(config.Arch.LSTM.h_size),
                                              str(config.Arch.Dense.hidden_size),
                                              box_info,
                                              player_id_info
                                              )

        saved_network = "{0}/oschulte/Galen/icehockey-models/lstm_Qs_model_saved_NN/" \
                        "{1}lstm_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            box_info,
            player_id_info
        )

    elif model_catagoery == 'lstm_win':
        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''
        log_dir = "{0}/oschulte/Galen/icehockey-models/lstm_win_log_NN" \
                  "/{1}lstm_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}".format(config.Learn.save_mother_dir,
                                              train_msg,
                                              str(config.Learn.feature_type),
                                              str(config.Learn.batch_size),
                                              str(config.Learn.iterate_num),
                                              str(config.Learn.learning_rate),
                                              str(config.Learn.model_type),
                                              str(config.Learn.max_seq_length),
                                              config.Learn.predict_target,
                                              None,
                                              str(config.Arch.LSTM.h_size),
                                              str(config.Arch.Dense.hidden_size),
                                              box_info,
                                              player_id_info
                                              )

        saved_network = "{0}/oschulte/Galen/icehockey-models/lstm_win_model_saved_NN/" \
                        "{1}lstm_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            box_info,
            player_id_info
        )

    elif model_catagoery == 'lstm_diff':

        if config.Learn.apply_pid:
            player_id_info = '_pid'
        else:
            player_id_info = ''

        log_dir = "{0}/oschulte/Galen/icehockey-models/lstm_diff_log_NN" \
                  "/{1}lstm_log_feature{2}_{8}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                  "_dense{11}{12}{13}".format(config.Learn.save_mother_dir,
                                              train_msg,
                                              str(config.Learn.feature_type),
                                              str(config.Learn.batch_size),
                                              str(config.Learn.iterate_num),
                                              str(config.Learn.learning_rate),
                                              str(config.Learn.model_type),
                                              str(config.Learn.max_seq_length),
                                              config.Learn.predict_target,
                                              None,
                                              str(config.Arch.LSTM.h_size),
                                              str(config.Arch.Dense.hidden_size),
                                              box_info,
                                              player_id_info
                                              )

        saved_network = "{0}/oschulte/Galen/icehockey-models/lstm_diff_model_saved_NN/" \
                        "{1}lstm_saved_networks_feature{2}_{8}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}" \
                        "_dense{11}{12}{13}".format(
            config.Learn.save_mother_dir,
            train_msg,
            str(config.Learn.feature_type),
            str(config.Learn.batch_size),
            str(config.Learn.iterate_num),
            str(config.Learn.learning_rate),
            str(config.Learn.model_type),
            str(config.Learn.max_seq_length),
            config.Learn.predict_target,
            None,
            str(config.Arch.LSTM.h_size),
            str(config.Arch.Dense.hidden_size),
            box_info,
            player_id_info
        )

    elif model_catagoery == 'cvae':  # TODO: add more parameters
        log_dir = "{0}/oschulte/Galen/icehockey-models/cvae_saved_NN" \
                  "/{1}cvae_log_feature{2}_latent{8}_x{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}{12}".format(config.Learn.save_mother_dir,
                                                              train_msg,
                                                              str(config.Learn.feature_type),
                                                              str(config.Learn.batch_size),
                                                              str(config.Learn.iterate_num),
                                                              str(config.Learn.learning_rate),
                                                              str(config.Learn.model_type),
                                                              None,
                                                              str(config.Arch.CVAE.latent_dim),
                                                              str(config.Arch.CVAE.x_dim),
                                                              str(config.Arch.CVAE.y_dim),
                                                              None,
                                                              box_info
                                                              )

        saved_network = "{0}/oschulte/Galen/icehockey-models/cvae_saved_NN/" \
                        "{1}cvae_saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}{12}".format(config.Learn.save_mother_dir,
                                                                    train_msg,
                                                                    str(config.Learn.feature_type),
                                                                    str(config.Learn.batch_size),
                                                                    str(config.Learn.iterate_num),
                                                                    str(config.Learn.learning_rate),
                                                                    str(config.Learn.model_type),
                                                                    None,
                                                                    str(config.Arch.CVAE.latent_dim),
                                                                    str(config.Arch.CVAE.x_dim),
                                                                    str(config.Arch.CVAE.y_dim),
                                                                    None,
                                                                    box_info
                                                                    )

    elif model_catagoery == 'encoder':

        if config.Learn.apply_lstm:
            lstm_msg = '_lstm'
        else:
            lstm_msg = ''

        log_dir = "{0}/oschulte/Galen/icehockey-models/stats_encoder_saved_NN" \
                  "/{1}encoder_log_feature{2}_embed{8}_in{9}_out{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}".format(config.Learn.save_mother_dir,
                                                                  train_msg,
                                                                  str(config.Learn.feature_type),
                                                                  str(config.Learn.batch_size),
                                                                  str(config.Learn.iterate_num),
                                                                  str(config.Learn.learning_rate),
                                                                  str(config.Learn.model_type),
                                                                  None,
                                                                  str(config.Arch.Encoder.embed_dim),
                                                                  str(config.Arch.Encoder.input_dim),
                                                                  str(config.Arch.Encoder.output_dim),
                                                                  None,
                                                                  box_info,
                                                                  lstm_msg
                                                                  )

        saved_network = "{0}/oschulte/Galen/icehockey-models/stats_encoder_saved_NN/" \
                        "{1}encoder_saved_networks_feature{2}_embed{8}_in{9}_out{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}{12}{13}".format(config.Learn.save_mother_dir,
                                                                        train_msg,
                                                                        str(config.Learn.feature_type),
                                                                        str(config.Learn.batch_size),
                                                                        str(config.Learn.iterate_num),
                                                                        str(config.Learn.learning_rate),
                                                                        str(config.Learn.model_type),
                                                                        None,
                                                                        str(config.Arch.Encoder.embed_dim),
                                                                        str(config.Arch.Encoder.input_dim),
                                                                        str(config.Arch.Encoder.output_dim),
                                                                        None,
                                                                        box_info,
                                                                        lstm_msg
                                                                        )

    return saved_network, log_dir


def compute_rnn_acc(target_actions_prob, output_actions_prob, selection_matrix, config, if_print=False):
    total_number = 0
    correct_number = 0
    correct_output_all = {}
    for batch_index in range(0, len(selection_matrix)):
        for trace_length_index in range(0, config.Learn.max_seq_length):
            if selection_matrix[batch_index][trace_length_index]:
                total_number += 1
                output_prediction = np.argmax(output_actions_prob[batch_index][trace_length_index])
                target_prediction = np.argmax(target_actions_prob[batch_index][trace_length_index])
                if output_prediction == target_prediction:
                    correct_number += 1
                    if correct_output_all.get(output_prediction) is None:
                        correct_output_all.update({output_prediction: 1})
                    else:
                        number = correct_output_all.get(output_prediction) + 1
                        correct_output_all.update({output_prediction: number})

    if if_print:
        print(correct_output_all)

    return float(correct_number) / float(total_number)


def compute_mae(target_actions_prob, output_actions_prob, if_print=False):
    total_number = 0
    total_mae = 0
    for batch_index in range(0, len(target_actions_prob)):
        total_number += 1
        mae = abs(output_actions_prob[batch_index] - target_actions_prob[batch_index])
        # print mae
        total_mae += mae
    print('prediction scale is :' + str(np.sum(output_actions_prob, axis=0)))
    return total_mae / float(total_number)


def compute_acc(target_actions_prob, output_actions_prob, if_print=False):
    total_number = 0
    correct_number = 0
    correct_output_all = {}
    for batch_index in range(0, len(target_actions_prob)):
        total_number += 1
        output_prediction = np.argmax(output_actions_prob[batch_index])
        target_prediction = np.argmax(target_actions_prob[batch_index])
        if output_prediction == target_prediction:
            correct_number += 1
            if correct_output_all.get(output_prediction) is None:
                correct_output_all.update({output_prediction: 1})
            else:
                number = correct_output_all.get(output_prediction) + 1
                correct_output_all.update({output_prediction: number})

    if if_print:
        print(correct_output_all)

    return float(correct_number) / float(total_number)


def calc_pdf(y, mu, var):
    """Calculate component density"""
    value = tf.subtract(y, mu) ** 2
    value = (1 / tf.sqrt(2 * np.pi * (var ** 2))) * tf.exp((-1 / (2 * (var ** 2))) * value)
    return value


def normal_td(mu1, mu2, var1, var2, y):
    """compute td error between two normal distribution"""
    mu_diff = (mu2 - mu1)
    var_diff = (var1 ** 2 + var2 ** 2) ** 0.5
    # https://stats.stackexchange.com/questions/186463/distribution-of-difference-between-two-normal-distributions
    com1 = (var_diff ** -1) * ((2 / np.pi) ** 0.5)
    com2 = tf.cosh(y * mu_diff / (var_diff ** 2))
    com3 = tf.exp(-1 * (y ** 2 + mu_diff ** 2) / (2 * var_diff ** 2))
    # return com1, com2, com3
    return com1 * com2 * com3


def compute_game_values(sess_nn, model, data_store, dir_game, config, player_id_cluster_dir, model_category):
    state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)

    action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                              max_length=config.Learn.max_seq_length)
    team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                               max_length=config.Learn.max_seq_length)
    player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)

    if model_category == "cvrnn":
        train_mask = np.asarray([[[0]] * config.Learn.max_seq_length] * len(player_index))
        if config.Learn.predict_target == 'PlayerLocalId':
            input_data = np.concatenate([player_index_seq,
                                         team_id_seq,
                                         state_input,
                                         action_seq,
                                         train_mask],
                                        axis=2)
            trace_lengths = state_trace_length
            selection_matrix_t0 = generate_selection_matrix(trace_lengths,
                                                            max_trace_length=config.Learn.max_seq_length)
        else:
            input_data = np.concatenate([player_index, state_input,
                                         action, train_mask], axis=2)
            trace_lengths = state_trace_length
            selection_matrix_t0 = generate_selection_matrix(trace_lengths,
                                                            max_trace_length=config.Learn.max_seq_length)

        [readout_next_Q, readout_accumu_Q] = sess_nn.run([model.sarsa_output,
                                                          model.diff_output],
                                                         feed_dict={model.input_data_ph: input_data,
                                                                    model.trace_length_ph: trace_lengths,
                                                                    model.selection_matrix_ph: selection_matrix_t0
                                                                    })
    elif model_category == 'lstm_Qs':
        input_data = np.concatenate([np.asarray(state_input), np.asarray(action_seq)], axis=2)
        trace_lengths = state_trace_length
        [readout_next_Q] = sess_nn.run([model.read_out],
                                       feed_dict={model.rnn_input_ph: input_data,
                                                  model.trace_lengths_ph: trace_lengths})
        readout_accumu_Q = None
    elif model_category == 'lstm_diff':
        if config.Learn.apply_pid:
            input_data = np.concatenate([np.asarray(player_index_seq),
                                         np.asarray(state_input), np.asarray(action_seq)], axis=2)
        else:
            input_data = np.concatenate([np.asarray(state_input), np.asarray(action_seq)], axis=2)
        trace_lengths = state_trace_length
        [readout_accumu_Q] = sess_nn.run([model.read_out],
                                         feed_dict={model.rnn_input_ph: input_data,
                                                    model.trace_lengths_ph: trace_lengths})
        readout_next_Q = None
    else:
        raise ValueError('unknown model {0}'.format(model_category))

    return readout_next_Q, readout_accumu_Q


def compute_games_Q_values(config, data_store_dir, dir_all,
                           model_number=None,
                           player_id_cluster_dir=None,
                           model_category=None,
                           return_values_flag=False):
    saved_network_dir, log_dir = get_model_and_log_name(config=config, model_catagoery=model_category)
    sess_nn = tf.InteractiveSession()
    if model_category == 'cvrnn':
        cvrnn = CVRNN(config=config)
        cvrnn()
        model_nn = cvrnn
        sess_nn.run(tf.global_variables_initializer())
        model_path = saved_network_dir + '/ice_hockey-2019-game--{0}'.format(model_number)
    elif model_category == 'lstm_Qs':
        model_nn = TD_Prediction(config=config)
        model_nn()
        sess_nn.run(tf.global_variables_initializer())
        model_path = saved_network_dir + '/Ice-Hockey-game--{0}'.format(model_number)
    elif model_category == 'lstm_diff':
        model_nn = Diff_Prediction(config=config)
        model_nn()
        sess_nn.run(tf.global_variables_initializer())
        model_path = saved_network_dir + '/Ice-Hockey-game--{0}'.format(model_number)
    else:
        raise ValueError('unknown model type {0}'.format(model_category))

    data_name = get_data_name(config=config, model_catagoery=model_category, model_number=model_number)
    if model_number is not None:
        saver = tf.train.Saver()
        saver.restore(sess_nn, model_path)
        print 'successfully load data from' + model_path
    else:
        raise ValueError('please provide a model number or no model will be loaded')

    model_next_Q_values_all = []
    model_accumu_Q_value_all = []
    for game_name_dir in dir_all:
        print('working for game {0}'.format(game_name_dir))
        game_name = game_name_dir.split('.')[0]
        # game_time_all = get_game_time(data_path, game_name_dir)
        readout_next_Q, readout_accumu_Q = compute_game_values(sess_nn=sess_nn,
                                                               model=model_nn,
                                                               data_store=data_store_dir,
                                                               dir_game=game_name,
                                                               config=config,
                                                               player_id_cluster_dir=player_id_cluster_dir,
                                                               model_category=model_category)
        # plot_game_Q_values(model_value)
        if readout_next_Q is not None:
            model_next_Q_value_json = {}
            for value_index in range(0, len(readout_next_Q)):
                model_next_Q_value_json.update({value_index: {'home': float(readout_next_Q[value_index][0]),
                                                              'away': float(readout_next_Q[value_index][1]),
                                                              'end': float(readout_next_Q[value_index][2])}})
            model_next_Q_values_all.append(model_next_Q_value_json)
        if readout_accumu_Q is not None:
            model_accumu_Q_value_json = {}
            for value_index in range(0, len(readout_accumu_Q)):
                model_accumu_Q_value_json.update({value_index: {'home': float(readout_accumu_Q[value_index][0]),
                                                                'away': float(readout_accumu_Q[value_index][1]),
                                                                'end': float(readout_accumu_Q[value_index][2])}})
            model_accumu_Q_value_all.append(model_accumu_Q_value_json)

        game_store_dir = game_name_dir.split('.')[0]
        if not return_values_flag:
            if readout_next_Q is not None:
                with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'next_Qs'),
                          'w') as outfile:
                    json.dump(model_next_Q_value_json, outfile)
            if readout_accumu_Q is not None:
                with open(data_store_dir + "/" + game_store_dir + "/" + data_name.replace('Qs', 'accumu_Qs'),
                          'w') as outfile:
                    json.dump(model_accumu_Q_value_json, outfile)

            # sio.savemat(data_store_dir + "/" + game_name_dir + "/" + data_name,
            #             {'model_value': np.asarray(model_value)})
    if return_values_flag:
        return model_next_Q_values_all, model_accumu_Q_value_all
    else:
        return data_name


if __name__ == '__main__':
    print('testing normal_td')
    sess = tf.Session()
    mu1 = tf.constant([0.480746])
    mu2 = tf.constant([0.47201255])
    var1 = tf.constant([0.3566948])
    var2 = tf.constant([0.35223854])
    y = tf.constant([0.1])
    # mu1 = tf.constant([[0.480746, 0.11928552, 0.3999685],
    #                    [0.47201255, 0.12002791, 0.40795958],
    #                    [0.48492602, 0.11869837, 0.39637566]])
    # mu2 = tf.constant([[0.47201255, 0.12002791, 0.40795958],
    #                    [0.48492602, 0.11869837, 0.39637566],
    #                    [0.4928479, 0.11631709, 0.39083505]])
    # var1 = tf.constant([[0.3566948, 0.69558066, 0.86941123],
    #                     [0.35223854, 0.6946951, 0.82976174],
    #                     [0.3467184, 0.68256944, 0.8444052]])
    # var2 = tf.constant([[0.35223854, 0.6946951, 0.82976174],
    #                     [0.3467184, 0.68256944, 0.8444052],
    #                     [0.34353644, 0.68014973, 0.8500393]])
    # y = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # y = y + 1e-10
    # print(sess.run(y))
    com1_tf, com2_tf, com3_tf = normal_td(mu1,
                                          mu2,
                                          var1,
                                          var2,
                                          y)
    com1, com2, com3 = sess.run([com1_tf, com2_tf, com3_tf])
    print(com1)
    print(com2)
    print(com3)
    print(com1 * com2 * com3)
