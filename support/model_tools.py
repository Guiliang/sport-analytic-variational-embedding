import tensorflow as tf


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
    log_dir = "{0}/oschulte/Galen/soccer-models/hybrid_sl_log_NN" \
              "/{1}Scale-tt-three-cut_together_log_feature{2}" \
              "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}".format(config.Learn.save_mother_dir,
                                                                  train_msg,
                                                                  str(config.Learn.feature_type),
                                                                  str(config.Learn.batch_size),
                                                                  str(config.Learn.iterate_num),
                                                                  str(config.Learn.learning_rate),
                                                                  str(config.Learn.model_type),
                                                                  str(config.Learn.max_seq_length))

    saved_network = "{0}/oschulte/Galen/soccer-models/hybrid_sl_saved_NN/" \
                    "{1}Scale-tt-three-cut_together_saved_networks_feature{2}" \
                    "_batch{3}_iterate{4}_lr{5}_{6}_MaxTL{7}".format(config.Learn.save_mother_dir,
                                                                        train_msg,
                                                                        str(config.Learn.feature_type),
                                                                        str(config.Learn.batch_size),
                                                                        str(config.Learn.iterate_num),
                                                                        str(config.Learn.learning_rate),
                                                                        str(config.Learn.model_type),
                                                                        str(config.Learn.max_seq_length))
    return saved_network, log_dir
