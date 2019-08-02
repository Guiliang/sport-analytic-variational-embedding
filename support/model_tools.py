import tensorflow as tf
import numpy as np


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


def get_model_and_log_name(config, train_flag=False):
    if train_flag:
        train_msg = 'Train_'
    else:
        train_msg = ''
    log_dir = "{0}/oschulte/Galen/icehockey-models/cvrnn_log_NN" \
              "/{1}cvrnn_log_feature{2}_latent{8}_x{9}_y{10}" \
              "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}".format(config.Learn.save_mother_dir,
                                                                        train_msg,
                                                                        str(config.Learn.feature_type),
                                                                        str(config.Learn.batch_size),
                                                                        str(config.Learn.iterate_num),
                                                                        str(config.Learn.learning_rate),
                                                                        str(config.Learn.model_type),
                                                                        str(config.Learn.max_seq_length),
                                                                        str(config.Arch.CVRNN.latent_dim),
                                                                        str(config.Arch.CVRNN.y_dim),
                                                                        str(config.Arch.CVRNN.x_dim),
                                                                        str(config.Arch.CVRNN.hidden_dim)
                                                                        )

    saved_network = "{0}/oschulte/Galen/icehockey-models/cvrnn_saved_NN/" \
                    "{1}cvrnn-saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{11}".format(config.Learn.save_mother_dir,
                                                                              train_msg,
                                                                              str(config.Learn.feature_type),
                                                                              str(config.Learn.batch_size),
                                                                              str(config.Learn.iterate_num),
                                                                              str(config.Learn.learning_rate),
                                                                              str(config.Learn.model_type),
                                                                              str(config.Learn.max_seq_length),
                                                                              str(config.Arch.CVRNN.latent_dim),
                                                                              str(config.Arch.CVRNN.y_dim),
                                                                              str(config.Arch.CVRNN.x_dim),
                                                                              str(config.Arch.CVRNN.hidden_dim)
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
