import tensorflow as tf
import numpy as np
import os
from support.data_processing_tools import transfer2seq, get_icehockey_game_data, get_together_training_batch
from nn_structure.lstm_prediction_nn import Td_Prediction_NN
from config.lstm_prediction_config import LSTMCongfig
from support.model_tools import compute_acc


def train_model(model, sess, config, input_data, target_data,
                trace_lengths, terminal):
    [
        output_prob,
        _
    ] = sess.run([
        model.read_out,
        model.train_op],
        feed_dict={model.rnn_input_ph: input_data,
                   model.y_ph: target_data,
                   model.trace_lengths_ph: trace_lengths}
    )
    acc = compute_acc(output_prob, target_data, if_print=False)
    print ("training acc is {0}".format(str(acc)))


def validation_model(testing_dir_games_all, data_store, config, sess, model):
    output_decoder_all = None
    target_data_all = None
    selection_matrix_all = None
    print('validating model')
    game_number = 0
    for dir_game in testing_dir_games_all:

        if dir_game == '.DS_Store':
            continue
        game_number += 1
        state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
            data_store=data_store, dir_game=dir_game, config=config)
        action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                  max_length=config.Learn.max_seq_length)
        team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                   max_length=config.Learn.max_seq_length)
        player_id_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                     max_length=config.Learn.max_seq_length)
        # print ("\n training file" + str(dir_game))
        # reward_count = sum(reward)
        # print ("reward number" + str(reward_count))
        if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
            raise Exception('state length does not equal to reward length')

        train_len = len(state_input)
        train_number = 0
        s_t0 = state_input[train_number]
        train_number += 1

        while True:
            # try:
            batch_return, \
            train_number, \
            s_tl, \
            print_flag = get_together_training_batch(s_t0=s_t0,
                                                     state_input=state_input,
                                                     reward=reward,
                                                     player_index=player_index,
                                                     train_number=train_number,
                                                     train_len=train_len,
                                                     state_trace_length=state_trace_length,
                                                     action=action_seq,
                                                     team_id=team_id_seq,
                                                     config=config)

            # get the batch variables
            # s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
            #                      team_id_t1, 0, 0
            s_t0_batch = [d[0] for d in batch_return]
            s_t1_batch = [d[1] for d in batch_return]
            r_t_batch = [d[2] for d in batch_return]
            trace_t0_batch = [d[3] for d in batch_return]
            trace_t1_batch = [d[4] for d in batch_return]
            action_id_t0 = [d[5] for d in batch_return]
            action_id_t1 = [d[6] for d in batch_return]
            team_id_t0_batch = [d[7] for d in batch_return]
            team_id_t1_batch = [d[8] for d in batch_return]
            player_id_t0_batch = [d[9] for d in batch_return]
            player_id_t1_batch = [d[10] for d in batch_return]

            input_data = np.concatenate([np.asarray(action_id_t0), np.asarray(s_t0_batch)], axis=2)
            target_data = np.asarray(player_id_t0_batch)
            trace_lengths = trace_t0_batch

            for i in range(0, len(batch_return)):
                terminal = batch_return[i][-2]
                # cut = batch_return[i][8]

            [
                output_prob
            ] = sess.run([
                model.read_out
            ],
                feed_dict={model.rnn_input_ph: input_data,
                           model.y_ph: target_data,
                           model.trace_lengths_ph: trace_lengths}
            )
            if output_decoder_all is None:
                output_decoder_all = output_prob
                target_data_all = target_data
            else:
                output_decoder_all = np.concatenate([output_decoder_all, output_prob], axis=0)
                target_data_all = np.concatenate([target_data_all, target_data], axis=0)
            s_t0 = s_tl
            if terminal:
                # save progress after a game
                # model.saver.save(sess, saved_network + '/' + config.learn.sport + '-game-',
                #                  global_step=game_number)
                # v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                # game_diff_record_dict.update({dir_game: v_diff_record_average})
                break

    acc = compute_acc(output_actions_prob=output_decoder_all, target_actions_prob=target_data_all, if_print=True)
    print ("validation acc is {0}".format(str(acc)))


def run_network(sess, model, config, training_dir_games_all, testing_dir_games_all, data_store):
    game_number = 0
    converge_flag = False

    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / config.Learn.number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= config.Learn.number_of_total_game * config.Learn.iterate_num:
            break
        else:
            converge_flag = True
        for dir_game in training_dir_games_all:
            if dir_game == '.DS_Store':
                continue
            game_number += 1
            game_cost_record = []
            state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
                data_store=data_store, dir_game=dir_game, config=config)
            action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                      max_length=config.Learn.max_seq_length)
            team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                       max_length=config.Learn.max_seq_length)
            player_id_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                         max_length=config.Learn.max_seq_length)
            print ("\n training file" + str(dir_game))
            # reward_count = sum(reward)
            # print ("reward number" + str(reward_count))
            if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state_input)
            train_number = 0
            s_t0 = state_input[train_number]
            train_number += 1

            while True:
                # try:
                batch_return, \
                train_number, \
                s_tl, \
                print_flag = get_together_training_batch(s_t0=s_t0,
                                                         state_input=state_input,
                                                         reward=reward,
                                                         player_index=player_index,
                                                         train_number=train_number,
                                                         train_len=train_len,
                                                         state_trace_length=state_trace_length,
                                                         action=action_seq,
                                                         team_id=team_id_seq,
                                                         config=config)

                # get the batch variables
                # s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, action_id_t0, action_id_t1, team_id_t0,
                #                      team_id_t1, 0, 0
                s_t0_batch = [d[0] for d in batch_return]
                s_t1_batch = [d[1] for d in batch_return]
                r_t_batch = [d[2] for d in batch_return]
                trace_t0_batch = [d[3] for d in batch_return]
                trace_t1_batch = [d[4] for d in batch_return]
                action_id_t0 = [d[5] for d in batch_return]
                action_id_t1 = [d[6] for d in batch_return]
                team_id_t0_batch = [d[7] for d in batch_return]
                team_id_t1_batch = [d[8] for d in batch_return]
                player_id_t0_batch = [d[9] for d in batch_return]
                player_id_t1_batch = [d[10] for d in batch_return]

                input_data = np.concatenate([np.asarray(action_id_t0), np.asarray(s_t0_batch)], axis=2)
                target_data = np.asarray(player_id_t0_batch)
                trace_lengths = trace_t0_batch

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][-2]
                    # cut = batch_return[i][8]

                pretrain_flag = True if game_number <= 10 else False
                train_model(model, sess, config, input_data, target_data,
                            trace_lengths, terminal)
                s_t0 = s_tl
                if terminal:
                    # save progress after a game
                    # model.saver.save(sess, saved_network + '/' + config.learn.sport + '-game-',
                    #                  global_step=game_number)
                    # v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    # game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break
            if game_number % 10 == 1:
                validation_model(testing_dir_games_all, data_store, config, sess, model)


def run():
    tt_lstm_config_path = "../ice_hockey_playerId_prediction.yaml"
    lstm_prediction_config = LSTMCongfig.load(tt_lstm_config_path)

    test_flag = False
    # saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_cvrnn_config)
    if test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = lstm_prediction_config.Learn.save_mother_dir+"/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 9]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: testing
    number_of_total_game = len(dir_games_all)
    lstm_prediction_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    model = Td_Prediction_NN(
        feature_number=lstm_prediction_config.Learn.feature_number,
        h_size=lstm_prediction_config.Arch.LSTM.h_size,
        max_trace_length=lstm_prediction_config.Learn.max_seq_length,
        learning_rate=lstm_prediction_config.Learn.learning_rate,
        embed_size=lstm_prediction_config.Learn.embed_size,
        output_layer_size=lstm_prediction_config.Learn.output_layer_size,
        lstm_layer_num=lstm_prediction_config.Arch.LSTM.lstm_layer_num,
        dense_layer_num=lstm_prediction_config.Learn.dense_layer_num,
        apply_softmax=lstm_prediction_config.Learn.apply_softmax)
    model.initialize_ph()
    model.build()
    model.call()
    sess.run(tf.global_variables_initializer())
    run_network(sess=sess, model=model, config=lstm_prediction_config,
                training_dir_games_all=training_dir_games_all, testing_dir_games_all=testing_dir_games_all,
                data_store=data_store_dir)
    sess.close()


if __name__ == '__main__':
    run()
