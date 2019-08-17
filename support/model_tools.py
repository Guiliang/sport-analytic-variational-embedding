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


def get_model_and_log_name(config, model_catagoery, train_flag=False):
    if train_flag:
        train_msg = 'Train_'
    else:
        train_msg = ''
    if model_catagoery == 'cvrnn':
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
                        "{1}cvrnn_saved_networks_feature{2}_latent{8}_x{9}_y{10}" \
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
    elif model_catagoery == 'de_embed':
        log_dir = "{0}/oschulte/Galen/icehockey-models/de_log_NN" \
                  "/{1}de_embed_log_feature{2}_{8}_embed{9}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}_dense{11}".format(config.Learn.save_mother_dir,
                                                                                      train_msg,
                                                                                      str(config.Learn.feature_type),
                                                                                      str(config.Learn.batch_size),
                                                                                      str(config.Learn.iterate_num),
                                                                                      str(config.Learn.learning_rate),
                                                                                      str(config.Learn.model_type),
                                                                                      str(config.Learn.max_seq_length),
                                                                                      config.Learn.predict_target,
                                                                                      str(
                                                                                          config.Arch.Encode.latent_size),
                                                                                      str(config.Arch.LSTM.h_size),
                                                                                      str(
                                                                                          config.Arch.Dense.hidden_node_size)
                                                                                      )

        saved_network = "{0}/oschulte/Galen/icehockey-models/de_model_saved_NN/" \
                        "{1}de_embed_saved_networks_feature{2}_{8}_embed{9}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}_dense{11}".format(
            config.Learn.save_mother_dir,
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
            str(config.Arch.Dense.hidden_node_size))
    elif model_catagoery == 'mdn_Qs':
        log_dir = "{0}/oschulte/Galen/icehockey-models/mdn_Qs_log_NN" \
                  "/{1}de_embed_log_feature{2}_{8}_y{10}" \
                  "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}_dense{11}".format(config.Learn.save_mother_dir,
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
                                                                                      str(
                                                                                          config.Arch.Dense.hidden_size)
                                                                                      )

        saved_network = "{0}/oschulte/Galen/icehockey-models/mdn_Qs_model_saved_NN/" \
                        "{1}de_embed_saved_networks_feature{2}_{8}_y{10}" \
                        "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}_LSTM{10}_dense{11}".format(
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
            str(config.Arch.Dense.hidden_size))

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
    return com1*com2*com3


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
