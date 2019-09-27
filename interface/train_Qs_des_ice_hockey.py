import sys
import traceback

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
from config.de_Qs_config import DEQsCongfig
from nn_structure.de_nn import TD_Prediction_Embed
from support.data_processing_tools import handle_trace_length, compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv
from support.data_processing_tools import get_icehockey_game_data, transfer2seq, generate_selection_matrix, \
    safely_expand_reward, generate_diff_player_cluster_id, q_values_output_mask
from support.model_tools import get_model_and_log_name, compute_rnn_acc
from support.plot_tools import plot_players_games


def gathering_running_and_run(dir_game, config, player_id_cluster_dir, data_store,
                              model, sess, training_flag, game_number, validate_cvrnn_flag=False,
                              validate_td_flag=False, validate_variance_flag=False,
                              output_decoder_all=None,
                              target_data_all=None, selection_matrix_all=None,
                              q_values_all=None):
    if validate_variance_flag:
        match_q_values_players_dict = {}
        for i in range(config.Learn.player_cluster_number):
            match_q_values_players_dict.update({i: []})
    else:
        match_q_values_players_dict = None

    state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)
    action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                              max_length=config.Learn.max_seq_length)
    team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                               max_length=config.Learn.max_seq_length)
    player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)
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
                                                 player_index=player_index_seq,
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

        player_embed_t0 = None  # TODO: insert player embedding
        input_data_t0 = np.concatenate([np.asarray(s_t0_batch), np.asarray(action_id_t0)], axis=2)
        trace_lengths_t0 = trace_t0_batch

        player_embed_t1 = None  # TODO: insert player embedding
        input_data_t1 = np.concatenate([np.asarray(s_t1_batch), np.asarray(action_id_t1)], axis=2)
        trace_lengths_t1 = trace_t1_batch

        for i in range(0, len(batch_return)):
            terminal = batch_return[i][-2]
            cut = batch_return[i][-1]

        if training_flag:
            train_td_model(model, sess, config, input_data_t0, trace_lengths_t0, player_embed_t0,
                           input_data_t1, trace_lengths_t1, player_embed_t1, r_t_batch, terminal, cut)

        else:
            validate_variance_flag = validate_variance_flag if train_number <= 500 else False
            q_values, match_q_values_players_dict = \
                td_validation(sess, model, trace_lengths_t0, player_embed_t0,
                              player_id_t0_batch, s_t0_batch, action_id_t0,
                              config, match_q_values_players_dict,
                              r_t_batch, terminal, cut, train_number, validate_variance_flag)

            if q_values_all is None:
                q_values_all = q_values
            else:
                q_values_all = np.concatenate([q_values_all, q_values], axis=0)

        s_t0 = s_tl
        if terminal:
            break

    return [output_decoder_all, target_data_all, selection_matrix_all, q_values_all, match_q_values_players_dict]


def run_network(sess, model, config, log_dir, save_network_dir,
                training_dir_games_all, testing_dir_games_all,
                data_store, player_id_cluster_dir, compute_embedding=True):
    game_number = 0
    converge_flag = False
    saver = tf.train.Saver(max_to_keep=300)

    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / config.Learn.number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= len(training_dir_games_all) * config.Learn.iterate_num:
            break
        # else:
        #     converge_flag = True
        for dir_game in training_dir_games_all:
            if dir_game == '.DS_Store':
                continue
            game_number += 1
            print ("\ntraining file" + str(dir_game))
            gathering_running_and_run(dir_game, config,
                                      player_id_cluster_dir, data_store, model, sess,
                                      training_flag=True, game_number=game_number)
            if game_number % 100 == 1:
                save_model(game_number, saver, sess, save_network_dir, config)
                validate_model(testing_dir_games_all, data_store, config,
                               sess, model, player_id_cluster_dir, train_game_number=game_number)


def save_model(game_number, saver, sess, save_network_dir, config):
    if (game_number - 1) % 300 == 0:
        # save progress after a game
        print 'saving game', game_number
        saver.save(sess, save_network_dir + '/' + config.Learn.data_name + '-game-',
                   global_step=game_number)


def train_td_model(model, sess, config, input_data_t0, trace_lengths_t0, player_embed_t0,
                   input_data_t1, trace_lengths_t1, player_embed_t1, r_t_batch, terminal, cut):
    [readout_t1_batch] = sess.run([model.sarsa_output],
                                  feed_dict={model.input_data_ph: input_data_t1,
                                             model.trace_length_ph: trace_lengths_t1,
                                             model.embed_ph: player_embed_t1
                                             })
    # r_t_batch = safely_expand_reward(reward_batch=r_t_batch, max_trace_length=config.Learn.max_seq_length)
    y_batch = []
    # print(len(r_t_batch))
    # print(np.sum(np.asarray(r_t_batch), axis=0))
    for i in range(0, len(r_t_batch)):

        if i == len(r_t_batch) - 1:
            if terminal or cut:
                y_home = float((r_t_batch[i])[0])
                y_away = float((r_t_batch[i])[1])
                y_end = float((r_t_batch[i])[2])
                y_batch.append([y_home, y_away, y_end])
                print([y_home, y_away, y_end])
                break
        y_home = float((r_t_batch[i])[0]) + config.Learn.gamma * \
                 ((readout_t1_batch[i]).tolist())[0]
        y_away = float((r_t_batch[i])[1]) + config.Learn.gamma * \
                 ((readout_t1_batch[i]).tolist())[1]
        y_end = float((r_t_batch[i])[2]) + config.Learn.gamma * \
                ((readout_t1_batch[i]).tolist())[2]
        y_batch.append([y_home, y_away, y_end])

    # perform gradient step
    y_batch = np.asarray(y_batch)
    [
        # sarsa_y_last,
        # z_encoder_last,
        # select_index,
        # z_encoder,
        avg_diff,
        _,
        readout
    ] = sess.run(
        [
            # model.sarsa_y_last,
            # model.z_encoder_last,
            # model.select_index,
            # model.z_encoder,
            model.td_avg_diff,
            model.train_yd_op,
            model.sarsa_output
        ],
        feed_dict={model.sarsa_target_ph: y_batch,
                   model.trace_length_ph: trace_lengths_t0,
                   model.input_data_ph: input_data_t0,
                   model.embed_ph: player_embed_t0
                   }
    )

    # print('avg diff:{0}, avg Qs:{1}'.format(avg_diff, str(np.mean(readout, axis=0))))


def td_validation(sess, model, trace_lengths_t0, player_embed_t0,
                  player_id_t0_batch, s_t0_batch, action_id_t0, config,
                  match_q_values_players_dict, r_t_batch, terminal, cut, train_number,
                  validate_variance_flag):
    if validate_variance_flag:
        all_player_id = generate_diff_player_cluster_id(player_id_t0_batch)

        # r_t_batch = safely_expand_reward(reward_batch=r_t_batch, max_trace_length=config.Learn.max_seq_length)
        for i in range(0, len(r_t_batch)):
            if i == len(r_t_batch) - 1:
                if terminal or cut:
                    y_home = float((r_t_batch[i])[0])
                    y_away = float((r_t_batch[i])[1])
                    y_end = float((r_t_batch[i])[2])
                    print ('reward {0} in train number {1}'.format(str([y_home, y_away, y_end]), str(train_number)))
                    break

        readout_var_all = []
        for index in range(0, len(all_player_id)):
            player_id_batch = all_player_id[index]  # generated player Id
            match_q_values = []
            input_data_var = np.concatenate([np.asarray(s_t0_batch),
                                             np.asarray(action_id_t0)], axis=2)
            [readout_var] = sess.run([model.sarsa_output],
                                     feed_dict={model.input_data_ph: input_data_var,
                                                model.trace_length_ph: trace_lengths_t0,
                                                model.embed_ph: player_embed_t0
                                                })
            for i in range(len(input_data_var)):
                match_q_values.append(readout_var[i])
                # match_q_values.append(readout_var[i * config.Learn.max_seq_length + trace_lengths_t0[i] - 1])
            match_q_values_player = match_q_values_players_dict.get(index)
            match_q_values_player += match_q_values
            match_q_values_players_dict.update({index: match_q_values_player})

            # readout_var_masked = q_values_output_mask(q_values=readout_var, trace_lengths=trace_lengths_t0,
            #                                           max_trace_length=config.Learn.max_seq_length)
            readout_var_all.append(readout_var)
        var_all = np.var(np.asarray(readout_var_all), axis=0)

        print('The mean of q values variance is {0}'.format(np.mean(var_all)))

    input_data_t0 = np.concatenate([np.asarray(player_id_t0_batch), np.asarray(s_t0_batch),
                                    np.asarray(action_id_t0), train_mask], axis=2)

    [readout] = sess.run([model.sarsa_output],
                         feed_dict={model.input_data_ph: input_data_t0,
                                    model.trace_length_ph: trace_lengths_t0,
                                    model.selection_matrix_ph: selection_matrix_t0
                                    })
    # readout_masked = q_values_output_mask(q_values=readout, trace_lengths=trace_lengths_t0,
    #                                       max_trace_length=config.Learn.max_seq_length)
    return readout, match_q_values_players_dict


def validate_model(testing_dir_games_all, data_store, config, sess, model,
                   player_id_cluster_dir, train_game_number, validate_cvrnn_flag=True, validate_td_flag=True):
    output_decoder_all = None
    target_data_all = None
    selection_matrix_all = None
    q_values_all = None
    validate_variance_flag = True
    print('validating model')
    for dir_game in testing_dir_games_all:

        if dir_game == '.DS_Store':
            continue

        [output_decoder_all,
         target_data_all,
         selection_matrix_all,
         q_values_all,
         match_q_values_players_dict] = gathering_running_and_run(dir_game, config,
                                                                  player_id_cluster_dir,
                                                                  data_store,
                                                                  model, sess,
                                                                  training_flag=False,
                                                                  game_number=None,
                                                                  validate_cvrnn_flag=True,
                                                                  validate_td_flag=True,
                                                                  validate_variance_flag=validate_variance_flag,
                                                                  output_decoder_all=output_decoder_all,
                                                                  target_data_all=target_data_all,
                                                                  selection_matrix_all=selection_matrix_all,
                                                                  q_values_all=q_values_all)
        validate_variance_flag = False
        if match_q_values_players_dict is not None:
            plot_players_games(match_q_values_players_dict, train_game_number)

    if validate_cvrnn_flag:
        acc = compute_rnn_acc(output_actions_prob=output_decoder_all, target_actions_prob=target_data_all,
                              selection_matrix=selection_matrix_all, config=config, if_print=True)
        print ("testing acc is {0}".format(str(acc)))
    if validate_td_flag:
        print ("testing avg qs is {0}".format(str(np.mean(q_values_all, axis=0))))


def run():
    local_test_flag = False
    icehockey_de_Qs_config_path = "../environment_settings/ice_hockey_predict_Qs_de.yaml"
    icehockey_de_Qs_config = DEQsCongfig.load(icehockey_de_Qs_config_path)
    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_de_Qs_config, model_catagoery='de_Qs')

    if local_test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = icehockey_de_Qs_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 9]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: testing
    number_of_total_game = len(dir_games_all)
    icehockey_de_Qs_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    model = TD_Prediction_Embed(config=icehockey_de_Qs_config)
    sess.run(tf.global_variables_initializer())


    if not local_test_flag:
        # save the training and testing dir list
        if not os.path.exists(saved_network_dir):
            os.mkdir(saved_network_dir)
        # save the training and testing dir list
        if os.path.exists(saved_network_dir + '/training_file_dirs_all.csv'):
            os.rename(saved_network_dir + '/training_file_dirs_all.csv',
                      saved_network_dir + '/bak_training_file_dirs_all.csv')
        if os.path.exists(saved_network_dir + '/testing_file_dirs_all.csv'):
            os.rename(saved_network_dir + '/testing_file_dirs_all.csv',
                      saved_network_dir + '/bak_testing_file_dirs_all.csv')
        # save the training and testing dir list
        with open(saved_network_dir + '/training_file_dirs_all.csv', 'wb') as f:
            for dir in dir_games_all[0: len(dir_games_all) / 10 * 8]:
                f.write(dir + '\n')
        with open(saved_network_dir + '/testing_file_dirs_all.csv', 'wb') as f:
            for dir in dir_games_all[len(dir_games_all) / 10 * 9:]:
                f.write(dir + '\n')
    run_network(sess=sess, model=model, config=icehockey_de_Qs_config, log_dir=log_dir,
                save_network_dir=saved_network_dir, data_store=data_store_dir,
                training_dir_games_all=training_dir_games_all, testing_dir_games_all=testing_dir_games_all,
                player_id_cluster_dir=None)
    sess.close()


if __name__ == '__main__':
    run()
