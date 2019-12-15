import csv
import datetime
import sys
import traceback

# from random import shuffle

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
from support.model_tools import ExperienceReplayBuffer, compute_acc, BalanceExperienceReplayBuffer
from config.cvae_config import CVAECongfig
from nn_structure.cvae_nn import CVAE_NN
from support.data_processing_tools import handle_trace_length, compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv, compute_game_win_vec, compute_game_score_diff_vec, \
    read_feature_within_events
from support.data_processing_tools import get_icehockey_game_data, transfer2seq, generate_selection_matrix, \
    safely_expand_reward, generate_diff_player_cluster_id, q_values_output_mask
from support.model_tools import get_model_and_log_name, compute_rnn_acc

# from support.plot_tools import plot_players_games

MemoryBuffer = ExperienceReplayBuffer(capacity_number=30000)
Prediction_MemoryBuffer = BalanceExperienceReplayBuffer(capacity_number=30000)


def gathering_running_and_run(dir_game, config, player_id_cluster_dir, data_store, source_data_dir,
                              model, sess, training_flag, game_number,
                              validate_cvrnn_flag=False,
                              validate_td_flag=False,
                              validate_diff_flag=False,
                              validate_variance_flag=False,
                              validate_predict_flag=False,
                              output_decoder_all=None,
                              target_data_all=None,
                              q_values_all=None,
                              output_label_all=None,
                              real_label_all=None,
                              pred_target_data_all=None,
                              pred_output_prob_all=None):
    if validate_variance_flag:
        match_q_values_players_dict = {}
        for i in range(config.Learn.player_cluster_number):
            match_q_values_players_dict.update({i: []})
    else:
        match_q_values_players_dict = None

    state_trace_length, state_input, reward, actions, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)

    if config.Learn.apply_lstm:
        # raise ValueError('no! do not use LSTM!')
        encoder_trace = state_trace_length
        encoder_state_input = state_input
        player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                        max_length=config.Learn.max_seq_length)
        encoder_player_index = player_index_seq
        action_seq = transfer2seq(data=actions, trace_length=state_trace_length,
                                  max_length=config.Learn.max_seq_length)
        encoder_action = action_seq
        team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                   max_length=config.Learn.max_seq_length)
        encoder_team_id = team_id_seq
        axis_concat = 2

    else:
        state_zero_trace = [1] * len(state_trace_length)
        state_zero_input = []
        for trace_index in range(0, len(state_trace_length)):
            trace_length = state_trace_length[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            state_zero_input.append(state_input[trace_index, trace_length, :])
        state_zero_input = np.asarray(state_zero_input)
        encoder_trace = state_zero_trace
        encoder_state_input = state_zero_input
        encoder_player_index = player_index
        encoder_action = actions
        encoder_team_id = team_id
        axis_concat = 1

    score_diff = compute_game_score_diff_vec(rewards=reward)
    score_difference_game = read_feature_within_events(dir_game,
                                                       source_data_dir,
                                                       'scoreDifferential',
                                                       transfer_home_number=True,
                                                       data_store=data_store)

    if config.Arch.Predict.predict_target == 'ActionGoal':
        add_pred_flag = True
        actions_all = read_feature_within_events(directory=dir_game,
                                                 data_path=source_data_dir,
                                                 feature_name='name')
        next_goal_label = []
        data_length = state_trace_length.shape[0]
        new_reward = []
        new_action = []
        new_state_input = []
        new_state_trace = []
        new_team_id = []
        new_player_index = []
        for action_index in range(0, data_length):
            action = actions_all[action_index]
            if 'shot' in action:
                if action_index + 1 == data_length:
                    continue
                if config.Learn.apply_lstm:
                    new_reward.append(reward[action_index])
                    new_action.append(encoder_action[action_index])
                    new_state_input.append(encoder_state_input[action_index])
                    new_state_trace.append(encoder_trace[action_index])
                    new_team_id.append(encoder_team_id[action_index])
                    new_player_index.append(encoder_player_index[action_index])
                else:
                    new_reward.append(reward[action_index])
                    new_action.append(encoder_action[action_index])
                    new_state_input.append(encoder_state_input[action_index])
                    new_state_trace.append(encoder_trace[action_index])
                    new_team_id.append(encoder_team_id[action_index])
                    new_player_index.append(encoder_player_index[action_index])
                if actions_all[action_index + 1] == 'goal':
                    # print(actions_all[action_index+1])
                    next_goal_label.append([1, 0])
                else:
                    # print(actions_all[action_index + 1])
                    next_goal_label.append([0, 1])
        pred_target = next_goal_label
    elif config.Arch.Predict.predict_target == 'Action':
        add_pred_flag = True
        pred_target = actions[1:, :]
        new_reward = reward[:-1]
        new_action = actions[:-1, :, :]
        new_state_input = encoder_state_input[:-1, :, :]
        new_state_trace = encoder_trace[:-1]
        new_team_id = team_id[:-1, :, :]
        new_player_index = player_index[:-1, :, :]
    else:
        # raise ValueError()
        add_pred_flag = False

    if add_pred_flag:
        if training_flag:
            pred_train_mask = np.asarray([1] * len(new_state_input))
        else:
            pred_train_mask = np.asarray([1] * len(new_state_input))  # we should apply encoder output
        if config.Learn.predict_target == 'PlayerLocalId':
            pred_input_data = np.concatenate([np.asarray(new_player_index),
                                              np.asarray(new_team_id),
                                              np.asarray(new_state_input),
                                              np.asarray(new_action), ], axis=axis_concat)
            pred_target_data = np.asarray(np.asarray(pred_target))
            pred_trace_lengths = new_state_trace
        else:
            pred_input_data = np.concatenate([np.asarray(new_player_index),
                                              np.asarray(new_state_input),
                                              np.asarray(new_action)], axis=axis_concat)
            pred_target_data = np.asarray(np.asarray(pred_target))
            pred_trace_lengths = new_state_trace
        if training_flag:
            for i in range(len(new_state_input)):
                cache_label = np.argmax(pred_target_data[i], axis=0)
                Prediction_MemoryBuffer.push([pred_input_data[i],
                                              pred_target_data[i],
                                              pred_trace_lengths[i],
                                              pred_train_mask[i]
                                              ], cache_label)

    # reward_count = sum(reward)
    # print ("reward number" + str(reward_count))
    if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
        raise Exception('state length does not equal to reward length')

    train_len = len(state_input)
    train_number = 0
    s_t0 = encoder_state_input[train_number]
    train_number += 1
    while True:
        # try:
        batch_return, \
        train_number, \
        s_tl, \
        print_flag = get_together_training_batch(s_t0=s_t0,
                                                 state_input=encoder_state_input,
                                                 reward=reward,
                                                 player_index=encoder_player_index,
                                                 train_number=train_number,
                                                 train_len=train_len,
                                                 state_trace_length=encoder_trace,
                                                 action=encoder_action,
                                                 team_id=encoder_team_id,
                                                 win_info=score_diff,
                                                 score_info=score_difference_game,
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
        win_id_t_batch = [d[11] for d in batch_return]
        terminal_batch = [d[-2] for d in batch_return]
        cut_batch = [d[-1] for d in batch_return]
        score_diff_t_batch = [d[11] for d in batch_return]
        score_diff_base_t0_batch = [d[12] for d in batch_return]
        outcome_data = score_diff_t_batch
        score_diff_base_t0 = score_diff_base_t0_batch
        # (player_id, state ,action flag)

        if training_flag:
            train_mask = np.asarray([1] * len(s_t0_batch))
        else:
            train_mask = np.asarray([0] * len(s_t0_batch))

        for i in range(0, len(terminal_batch)):
            terminal = terminal_batch[i]
            cut = cut_batch[i]
        if config.Learn.predict_target == 'PlayerLocalId':
            input_data_t0 = np.concatenate([np.asarray(player_id_t0_batch),
                                            np.asarray(team_id_t0_batch),
                                            np.asarray(s_t0_batch),
                                            np.asarray(action_id_t0)], axis=axis_concat)
            target_data_t0 = np.asarray(np.asarray(player_id_t0_batch))
            input_data_t1 = np.concatenate([np.asarray(player_id_t1_batch),
                                            np.asarray(team_id_t1_batch),
                                            np.asarray(s_t1_batch),
                                            np.asarray(action_id_t1)], axis=axis_concat)
            target_data_t1 = np.asarray(np.asarray(player_id_t1_batch))
            trace_length_t0 = np.asarray(trace_t0_batch)
            trace_length_t1 = np.asarray(trace_t1_batch)
        else:
            input_data_t0 = np.concatenate([np.asarray(player_id_t0_batch), np.asarray(s_t0_batch),
                                            np.asarray(action_id_t0)], axis=axis_concat)
            target_data_t0 = np.asarray(np.asarray(player_id_t0_batch))
            input_data_t1 = np.concatenate([np.asarray(player_id_t1_batch), np.asarray(s_t1_batch),
                                            np.asarray(action_id_t1)], axis=axis_concat)
            target_data_t1 = np.asarray(np.asarray(player_id_t1_batch))
            trace_length_t0 = np.asarray(trace_t0_batch)
            trace_length_t1 = np.asarray(trace_t1_batch)

        if training_flag:

            if config.Learn.apply_stochastic:
                for i in range(len(input_data_t0)):
                    MemoryBuffer.push([input_data_t0[i], target_data_t0[i], trace_length_t0[i],
                                       input_data_t1[i], target_data_t1[i], trace_length_t1[i],
                                       r_t_batch[i], win_id_t_batch[i],
                                       terminal_batch[i], cut_batch[i]
                                       ])
                sampled_data = MemoryBuffer.sample(batch_size=config.Learn.batch_size)
                sample_input_data_t0 = np.asarray([sampled_data[j][0] for j in range(len(sampled_data))])
                sample_target_data_t0 = np.asarray([sampled_data[j][1] for j in range(len(sampled_data))])
                sample_trace_length_t0 = np.asarray([sampled_data[j][2] for j in range(len(sampled_data))])
                if training_flag:
                    train_mask = np.asarray([1] * len(sample_input_data_t0))
                else:
                    train_mask = np.asarray([0] * len(sample_input_data_t0))

                for i in range(0, len(terminal_batch)):
                    batch_terminal = terminal_batch[i]
                    batch_cut = cut_batch[i]
            else:
                sample_input_data_t0 = input_data_t0
                sample_target_data_t0 = target_data_t0
                sample_trace_length_t0 = trace_length_t0
                sampled_outcome_t = win_id_t_batch

            likelihood_loss, kl_loss = train_cvae_model(model, sess, config, sample_input_data_t0,
                                                        sample_target_data_t0, sample_trace_length_t0, train_mask)
            if terminal or cut:
                print('\n kl loss is {0} while ll lost is {1}'.format(str(kl_loss), str(likelihood_loss)))

            """we skip sampling for TD learning"""
            train_td_model(model, sess, config, input_data_t0, trace_length_t0,
                           input_data_t1, trace_length_t1, r_t_batch, train_mask, terminal, cut)

            train_score_diff(model, sess, config, input_data_t0, trace_length_t0,
                             input_data_t1, trace_length_t1, r_t_batch, outcome_data,
                             score_diff_base_t0, train_mask, terminal, cut)

            if add_pred_flag:
                sampled_data = Prediction_MemoryBuffer.sample(batch_size=config.Learn.batch_size)
                sample_pred_input_data = np.asarray([sampled_data[j][0] for j in range(len(sampled_data))])
                sample_pred_target_data = np.asarray([sampled_data[j][1] for j in range(len(sampled_data))])
                sample_pred_trace_lengths = np.asarray([sampled_data[j][2] for j in range(len(sampled_data))])
                sample_pred_train_mask = np.asarray([sampled_data[j][3] for j in range(len(sampled_data))])

                train_prediction(model, sess, config,
                                 sample_pred_input_data,
                                 sample_pred_target_data,
                                 sample_pred_trace_lengths,
                                 sample_pred_train_mask)

        else:
            # for i in range(0, len(r_t_batch)):
            #     if i == len(r_t_batch) - 1:
            #         if terminal or cut:
            #             print(r_t_batch[i])
            if validate_cvrnn_flag:
                output_decoder = cvae_validation(sess, model,
                                                 input_data_t0, target_data_t0,
                                                 trace_length_t0,
                                                 train_mask, config)

                if output_decoder_all is None:
                    output_decoder_all = output_decoder
                    target_data_all = target_data_t0
                else:
                    # try:
                    output_decoder_all = np.concatenate([output_decoder_all, output_decoder], axis=0)
                    target_data_all = np.concatenate([target_data_all, target_data_t0], axis=0)

            if validate_td_flag:
                # validate_variance_flag = validate_variance_flag if train_number <= 500 else False
                q_values = td_validation(sess, model, input_data_t0, trace_length_t0, train_mask, config)

                if q_values_all is None:
                    q_values_all = q_values
                else:
                    q_values_all = np.concatenate([q_values_all, q_values], axis=0)

            if validate_diff_flag:
                output_label, real_label = diff_validation(sess, model, input_data_t0, trace_length_t0,
                                                           train_mask,
                                                           score_diff_base_t0,
                                                           config, outcome_data)
                if real_label_all is None:
                    real_label_all = real_label
                else:
                    real_label_all = np.concatenate([real_label_all, real_label], axis=0)

                if output_label_all is None:
                    output_label_all = output_label
                else:
                    output_label_all = np.concatenate([output_label_all, output_label], axis=0)

        s_t0 = s_tl
        if terminal:
            break

    if validate_predict_flag and add_pred_flag:
        pred_target_data, pred_output_prob = validate_prediction(model, sess, config,
                                                                 pred_input_data,
                                                                 pred_target_data,
                                                                 pred_trace_lengths,
                                                                 pred_train_mask)
        if pred_target_data_all is None:
            pred_target_data_all = pred_target_data
        else:
            pred_target_data_all = np.concatenate([pred_target_data_all, pred_target_data], axis=0)

        if pred_output_prob_all is None:
            pred_output_prob_all = pred_output_prob
        else:
            pred_output_prob_all = np.concatenate([pred_output_prob_all, pred_output_prob], axis=0)

    return [output_decoder_all, target_data_all,
            q_values_all, real_label_all, output_label_all, pred_target_data_all,
            pred_output_prob_all]


def run_network(sess, model, config, log_dir, save_network_dir,
                training_dir_games_all, testing_dir_games_all,
                data_store, source_data_dir, player_id_cluster_dir, save_flag=False):
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
                                      player_id_cluster_dir, data_store, source_data_dir, model, sess,
                                      training_flag=True, game_number=game_number)
            save_model(game_number, saver, sess, save_network_dir, config)
            if game_number % 100 == 1:
                validate_model(testing_dir_games_all, data_store, source_data_dir, config,
                               sess, model, player_id_cluster_dir,
                               train_game_number=game_number,
                               validate_cvrnn_flag=True,
                               validate_td_flag=True,
                               validate_diff_flag=True,
                               validate_predict_flag=True)


def save_model(game_number, saver, sess, save_network_dir, config):
    if (game_number - 1) % 30 == 0:
        # save progress after a game
        print 'saving game', game_number
        saver.save(sess, save_network_dir + '/' + config.Learn.data_name + '-game-',
                   global_step=game_number)


# def train_win_prob(model, sess, config, input_data, trace_lengths, selection_matrix, outcome_data):
#     [
#         _,
#         win_output
#     ] = sess.run([
#         model.train_win_op,
#         model.win_output
#     ],
#         feed_dict={model.input_data_ph: input_data,
#                    model.win_target_ph: outcome_data,
#                    model.trace_length_ph: trace_lengths,
#                    model.selection_matrix_ph: selection_matrix
#                    })
#
#     output_label = np.argmax(win_output, axis=1)
#     real_label = np.argmax(outcome_data, axis=1)
#
#     correct_num = 0
#     for index in range(0, len(input_data)):
#         if output_label[index] == real_label[index]:
#             correct_num += 1
#
#     # print('accuracy of win prob is {0}'.format(str(float(correct_num)/len(input_data))))


def train_score_diff(model, sess, config,
                     input_data_t0, trace_length_t0,
                     input_data_t1, trace_length_t1,
                     reward_t, outcome_data,
                     score_diff_base_t0, train_mask, terminal, cut):
    if config.Learn.apply_lstm:
        x_ph_input_t1 = []
        for trace_index in range(0, len(trace_length_t1)):
            trace_length = trace_length_t1[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input_t1.append(input_data_t1[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input_t1 = np.asarray(x_ph_input_t1)
        feed_dict_1 = {model.x_ph: x_ph_input_t1,
                       model.y_ph: input_data_t1[:, :, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       model.trace_lengths_ph: trace_length_t1
                       }
    else:
        feed_dict_1 = {model.x_ph: input_data_t1[:, : config.Arch.CVAE.x_dim],
                       model.y_ph: input_data_t1[:, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       }

    [readout_t1_batch] = sess.run([model.q_values_diff],
                                  feed_dict=feed_dict_1)
    y_batch = []
    for i in range(0, len(readout_t1_batch)):
        if cut and i == len(readout_t1_batch) - 1:
            y_home = readout_t1_batch[i].tolist()[0] + float(reward_t[i][0])
            y_away = readout_t1_batch[i].tolist()[1] + float(reward_t[i][1])
            y_end = readout_t1_batch[i].tolist()[2] + float(reward_t[i][2])
            y_batch.append([y_home, y_away, y_end])
            # print('The accumulative Q values are '+str([y_home, y_away, y_end]))
            break
        # if terminal, only equals reward
        if terminal and i == len(readout_t1_batch) - 1:
            y_home = float(reward_t[i][0])
            y_away = float(reward_t[i][1])
            y_end = float(reward_t[i][2])
            # print('The accumulative Q values are '+str([y_home, y_away, y_end]))
            print('game is ending with {0}'.format(str([y_home, y_away, y_end])))
            y_batch.append([y_home, y_away, y_end])
            break
        else:
            y_home = readout_t1_batch[i].tolist()[0] + float(reward_t[i][0])
            y_away = readout_t1_batch[i].tolist()[1] + float(reward_t[i][1])
            y_end = readout_t1_batch[i].tolist()[2] + float(reward_t[i][2])
            y_batch.append([y_home, y_away, y_end])

    train_list = [model.q_values_diff, model.td_score_diff_diff, model.train_diff_op]

    if config.Learn.apply_lstm:
        x_ph_input_t0 = []
        for trace_index in range(0, len(trace_length_t0)):
            trace_length = trace_length_t0[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input_t0.append(input_data_t0[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input_t0 = np.asarray(x_ph_input_t0)
        feed_dict_0 = {model.x_ph: x_ph_input_t0,
                       model.y_ph: input_data_t0[:, :, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       model.trace_lengths_ph: trace_length_t0,
                       model.score_diff_target_ph: y_batch,
                       }
    else:
        feed_dict_0 = {model.x_ph: input_data_t0[:, : config.Arch.CVAE.x_dim],
                       model.y_ph: input_data_t0[:, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       model.score_diff_target_ph: y_batch,
                       }

    train_outputs = \
        sess.run(
            train_list,
            feed_dict=feed_dict_0
        )
    if terminal or cut:
        print('the avg diff Q values are home {0}, away {1} '
              'and end {2} with diff {3}'.format(np.mean(train_outputs[0][:, 0]),
                                                 np.mean(train_outputs[0][:, 1]),
                                                 np.mean(train_outputs[0][:, 2]),
                                                 np.mean(train_outputs[1])))
    output_label = train_outputs[0][:, 0] - train_outputs[0][:, 1] + np.asarray(score_diff_base_t0)
    real_label = outcome_data

    return output_label, real_label


def train_td_model(model, sess, config,
                   input_data_t0, trace_length_t0,
                   input_data_t1, trace_length_t1,
                   r_t_batch, train_mask, terminal, cut):
    if config.Learn.apply_lstm:
        x_ph_input_t1 = []
        for trace_index in range(0, len(trace_length_t1)):
            trace_length = trace_length_t1[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input_t1.append(input_data_t1[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input_t1 = np.asarray(x_ph_input_t1)
        feed_dict_1 = {model.x_ph: x_ph_input_t1,
                       model.y_ph: input_data_t1[:, :, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       model.trace_lengths_ph: trace_length_t1
                       }
    else:
        feed_dict_1 = {model.x_ph: input_data_t1[:, : config.Arch.CVAE.x_dim],
                       model.y_ph: input_data_t1[:, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       }

    [readout_t1_batch] = sess.run([model.q_values_sarsa],
                                  feed_dict=feed_dict_1)
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

    if config.Learn.apply_lstm:
        x_ph_input_t0 = []
        for trace_index in range(0, len(trace_length_t0)):
            trace_length = trace_length_t0[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input_t0.append(input_data_t0[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input_t0 = np.asarray(x_ph_input_t0)
        feed_dict_0 = {model.x_ph: x_ph_input_t0,
                       model.y_ph: input_data_t0[:, :, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       model.trace_lengths_ph: trace_length_t0,
                       model.sarsa_target_ph: y_batch,
                       }
    else:
        feed_dict_0 = {model.x_ph: input_data_t0[:, : config.Arch.CVAE.x_dim],
                       model.y_ph: input_data_t0[:, config.Arch.CVAE.x_dim:],
                       model.train_flag_ph: train_mask,
                       model.sarsa_target_ph: y_batch,
                       }

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
            model.train_td_op,
            model.q_values_sarsa
        ],
        feed_dict=feed_dict_0
    )

    if terminal or cut:
        print('the avg next goal Q values are home {0}, away {1} '
              'and end {2} with diff {3}'.format(np.mean(readout[:, 0]),
                                                 np.mean(readout[:, 1]),
                                                 np.mean(readout[:, 2]),
                                                 np.mean(avg_diff)))


def train_cvae_model(model, sess, config, input_data, target_data, trace_lengths, train_mask,
                     pretrain_flag=False):
    if config.Learn.apply_lstm:
        x_ph_input = []
        for trace_index in range(0, len(trace_lengths)):
            trace_length = trace_lengths[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input.append(input_data[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input = np.asarray(x_ph_input)

        feed_dict = {model.x_ph: x_ph_input,
                     model.y_ph: input_data[:, :, config.Arch.CVAE.x_dim:],
                     model.trace_lengths_ph: trace_lengths,
                     model.train_flag_ph: train_mask,
                     }
    else:
        feed_dict = {model.x_ph: input_data[:, : config.Arch.CVAE.x_dim],
                     model.y_ph: input_data[:, config.Arch.CVAE.x_dim:],
                     model.train_flag_ph: train_mask,
                     }

    [
        output_x,
        kl_loss,
        likelihood_loss,
        _
    ] = sess.run([
        model.x_,
        # model.cost,
        model.KL_divergence_loss,
        model.marginal_likelihood_loss,
        model.train_cvae_op],
        feed_dict=feed_dict
    )

    acc = compute_acc(target_label=target_data, output_prob=output_x)
    # print acc
    # if cost_out > 0.0001: # TODO: we still need to consider how to define convergence
    #     converge_flag = False
    # print('kl loss is {0} while ll lost is {1}'.format(str(kl_loss), str(likelihood_loss)))
    return likelihood_loss, kl_loss


def validate_prediction(model, sess, config,
                        pred_input_data,
                        pred_target_data,
                        pred_trace_lengths,
                        pred_train_mask):
    if config.Learn.apply_lstm:
        x_ph_input = []
        for trace_index in range(0, len(pred_trace_lengths)):
            trace_length = pred_trace_lengths[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input.append(pred_input_data[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input = np.asarray(x_ph_input)

        feed_dict = {model.x_ph: x_ph_input,
                     model.y_ph: pred_input_data[:, :, config.Arch.CVAE.x_dim:],
                     # model.predict_target_ph: pred_target_data,
                     model.trace_lengths_ph: pred_trace_lengths,
                     model.train_flag_ph: pred_train_mask,
                     }
    else:
        feed_dict = {model.x_ph: pred_input_data[:, : config.Arch.CVAE.x_dim],
                     # model.predict_target_ph: pred_target_data,
                     model.train_flag_ph: pred_train_mask,
                     model.y_ph: pred_input_data[:, config.Arch.CVAE.x_dim:]
                     }

    [
        predict_output,
    ] = sess.run([
        model.predict_output,
    ]
        ,
        feed_dict=feed_dict
    )

    # acc = compute_acc(predict_output, sample_pred_target_data, if_print=False)
    return predict_output, pred_target_data


def train_prediction(model, sess, config,
                     sample_pred_input_data,
                     sample_pred_target_data,
                     sample_pred_trace_lengths,
                     sample_pred_train_mask):
    if config.Learn.apply_lstm:
        x_ph_input = []
        for trace_index in range(0, len(sample_pred_trace_lengths)):
            trace_length = sample_pred_trace_lengths[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input.append(sample_pred_input_data[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input = np.asarray(x_ph_input)

        feed_dict = {model.x_ph: x_ph_input,
                     model.y_ph: sample_pred_input_data[:, :, config.Arch.CVAE.x_dim:],
                     model.predict_target_ph: sample_pred_target_data,
                     model.trace_lengths_ph: sample_pred_trace_lengths,
                     model.train_flag_ph: sample_pred_train_mask,
                     }
    else:
        feed_dict = {model.x_ph: sample_pred_input_data[:, : config.Arch.CVAE.x_dim],
                     model.predict_target_ph: sample_pred_target_data,
                     model.train_flag_ph: sample_pred_train_mask,
                     model.y_ph: sample_pred_input_data[:, config.Arch.CVAE.x_dim:]
                     }

    [
        predict_output,
        _
    ] = sess.run([
        model.predict_output,
        model.train_predict_op],
        feed_dict=feed_dict
    )

    acc = compute_acc(predict_output, sample_pred_target_data, if_print=False)
    pass


def cvae_validation(sess, model, input_data_t0, target_data_t0, trace_length_t0, train_mask, config):

    if config.Learn.apply_lstm:
        x_ph_input = []
        for trace_index in range(0, len(trace_length_t0)):
            trace_length = trace_length_t0[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input.append(input_data_t0[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input = np.asarray(x_ph_input)

        feed_dict = {model.x_ph: x_ph_input,
                     model.y_ph: input_data_t0[:, :, config.Arch.CVAE.x_dim:],
                     model.trace_lengths_ph: trace_length_t0,
                     model.train_flag_ph: train_mask,
                     }
    else:
        feed_dict = {model.x_ph: input_data_t0[:, : config.Arch.CVAE.x_dim],
                     model.y_ph: input_data_t0[:, config.Arch.CVAE.x_dim:],
                     model.train_flag_ph: train_mask,
                     }

    [
        batch_size,
        output_x,
    ] = sess.run([
        model.batch_size,
        model.x_],
        feed_dict=feed_dict
    )
    # print(batch_size)
    return output_x


def td_validation(sess, model, input_data_t0, trace_length_t0, train_mask, config):
    if config.Learn.apply_lstm:
        x_ph_input = []
        for trace_index in range(0, len(trace_length_t0)):
            trace_length = trace_length_t0[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input.append(input_data_t0[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input = np.asarray(x_ph_input)
        feed_dict = {model.x_ph: x_ph_input,
                     model.y_ph: input_data_t0[:, :, config.Arch.CVAE.x_dim:],
                     model.train_flag_ph: train_mask,
                     model.trace_lengths_ph: trace_length_t0
                     }
    else:
        feed_dict = {model.x_ph: input_data_t0[:, : config.Arch.CVAE.x_dim],
                     model.y_ph: input_data_t0[:, config.Arch.CVAE.x_dim:],
                     model.train_flag_ph: train_mask,
                     }

    [readout] = sess.run([model.q_values_sarsa],
                         feed_dict=feed_dict)
    # readout_masked = q_values_output_mask(q_values=readout, trace_lengths=trace_lengths_t0,
    #                                       max_trace_length=config.Learn.max_seq_length)
    return readout


# def win_validation(sess, model, input_data,
#                    trace_lengths, selection_matrix,
#                    config, outcome_data):
#     [
#         # _,
#         win_output
#     ] = sess.run([
#         # model.train_win_op,
#         model.win_output],
#         feed_dict={model.input_data_ph: input_data,
#                    model.win_target_ph: outcome_data,
#                    model.trace_length_ph: trace_lengths,
#                    model.selection_matrix_ph: selection_matrix
#                    })
#     output_label = np.argmax(win_output, axis=1)
#     real_label = np.argmax(outcome_data, axis=1)
#
#     # correct_num = 0
#     # for index in range(0, len(input_data)):
#     #     if output_label[index] == real_label[index]:
#     #         correct_num += 1
#     #
#     return output_label, real_label


def diff_validation(sess, model,
                    input_data_t0,
                    trace_length_t0,
                    train_mask,
                    score_diff_base_t0,
                    config, outcome_data):
    if config.Learn.apply_lstm:
        x_ph_input = []
        for trace_index in range(0, len(trace_length_t0)):
            trace_length = trace_length_t0[trace_index]
            trace_length = trace_length - 1
            if trace_length > 9:
                trace_length = 9
            x_ph_input.append(input_data_t0[trace_index, trace_length, : config.Arch.CVAE.x_dim])
        x_ph_input = np.asarray(x_ph_input)
        feed_dict = {model.x_ph: x_ph_input,
                     model.y_ph: input_data_t0[:, :, config.Arch.CVAE.x_dim:],
                     model.train_flag_ph: train_mask,
                     model.trace_lengths_ph: trace_length_t0
                     }
    else:
        feed_dict = {model.x_ph: input_data_t0[:, : config.Arch.CVAE.x_dim],
                     model.y_ph: input_data_t0[:, config.Arch.CVAE.x_dim:],
                     model.train_flag_ph: train_mask,
                     }

    train_outputs = sess.run([model.q_values_diff],
                             feed_dict=feed_dict)
    if train_outputs[0].shape[0] > 1:
        output_label = train_outputs[0][:, 0] - train_outputs[0][:, 1] + np.asarray(score_diff_base_t0)
        real_label = outcome_data
    else:
        output_label = [train_outputs[0][0][0] - train_outputs[0][0][1] + score_diff_base_t0[0]]
        real_label = outcome_data

    return output_label, real_label


def validate_model(testing_dir_games_all, data_store, source_data_dir, config, sess, model,
                   player_id_cluster_dir, train_game_number, validate_cvrnn_flag, validate_td_flag,
                   validate_diff_flag, validate_predict_flag):
    output_decoder_all = None
    target_data_all = None
    q_values_all = None
    validate_variance_flag = False
    pred_target_data_all = None
    pred_output_prob_all = None

    if validate_diff_flag:
        real_label_record = np.ones([len(testing_dir_games_all), 5000]) * -1
        output_label_record = np.ones([len(testing_dir_games_all), 5000]) * -1

    print('validating model')
    for dir_index in range(0, len(testing_dir_games_all)):

        real_label_all = None
        output_label_all = None

        dir_game = testing_dir_games_all[dir_index]
        print('validating game {0}'.format(str(dir_game)))
        if dir_game == '.DS_Store':
            continue

        [output_decoder_all,
         target_data_all,
         q_values_all,
         real_label_all,
         output_label_all,
         pred_target_data_all,
         pred_output_prob_all] = gathering_running_and_run(dir_game, config,
                                                           player_id_cluster_dir,
                                                           data_store,
                                                           source_data_dir,
                                                           model, sess,
                                                           training_flag=False,
                                                           game_number=None,
                                                           validate_cvrnn_flag=validate_cvrnn_flag,
                                                           validate_td_flag=validate_td_flag,
                                                           validate_diff_flag=validate_diff_flag,
                                                           validate_variance_flag=validate_variance_flag,
                                                           validate_predict_flag=validate_predict_flag,
                                                           output_decoder_all=output_decoder_all,
                                                           target_data_all=target_data_all,
                                                           q_values_all=q_values_all,
                                                           output_label_all=output_label_all,
                                                           real_label_all=real_label_all,
                                                           pred_target_data_all=pred_target_data_all,
                                                           pred_output_prob_all=pred_output_prob_all)
        # validate_variance_flag = False
        # if match_q_values_players_dict is not None:
        #     plot_players_games(match_q_values_players_dict, train_game_number)

        if validate_diff_flag:
            real_label_record[dir_index][:len(real_label_all)] = real_label_all[:len(real_label_all)]
            output_label_record[dir_index][:len(output_label_all)] = output_label_all[:len(output_label_all)]

    if validate_cvrnn_flag:
        acc = compute_acc(target_data_all, output_decoder_all, if_print=True)
        print ("testing acc is {0}".format(str(acc)))
    if validate_td_flag:
        print ("testing avg qs is {0}".format(str(np.mean(q_values_all, axis=0))))

    if validate_diff_flag:
        # print ('general real label is {0}'.format(str(np.sum(real_label_record, axis=1))))
        # print ('general output label is {0}'.format(str(np.sum(output_label_record, axis=1))))
        for i in range(0, output_label_record.shape[1]):
            real_outcome_record_step = real_label_record[:, i]
            model_output_record_step = output_label_record[:, i]
            diff_sum = 0
            total_number = 0
            print_flag = True
            for win_index in range(0, len(real_outcome_record_step)):
                if model_output_record_step[win_index] == -100 or real_outcome_record_step[win_index] == -100:
                    print_flag = True
                    continue
                diff = abs(model_output_record_step[win_index] - real_outcome_record_step[win_index])
                diff_sum += diff
                total_number += 1
            if print_flag:
                if i % 100 == 0 and total_number > 0:
                    print('diff of time {0} is {1}'.format(str(i), str(float(diff_sum) / total_number)))
    if validate_predict_flag:
        acc = compute_acc(pred_output_prob_all, pred_target_data_all, if_print=True)
        print('prediction acc is {0}'.format(str(acc)))


def run():
    training = True
    local_test_flag = False
    box_msg = ''
    player_id_type = 'local_id'
    predict_action = '_predict_next_goal'
    rnn_type = '_lstm'
    model_type = 'vhe'

    if player_id_type == 'ap_cluster':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/player_id_ap_cluster.json'
        predicted_target = '_PlayerPositionClusterAP'  # playerId_
    elif player_id_type == 'km_cluster':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/player_id_km_cluster.json'
        predicted_target = '_PlayerPositionClusterKM'  # playerId_
    elif player_id_type == 'local_id':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        predicted_target = '_PlayerLocalId'  # playerId_
    else:
        player_id_cluster_dir = None
        predicted_target = ''

    icehockey_cvae_config_path = "../environment_settings/" \
                                 "icehockey_{4}{3}{0}{1}_config{2}.yaml".format(predicted_target,
                                                                                 predict_action,
                                                                                 box_msg,
                                                                                 rnn_type,
                                                                               model_type)
    icehockey_cvae_config = CVAECongfig.load(icehockey_cvae_config_path)
    Prediction_MemoryBuffer.set_cache_memory(cache_number=icehockey_cvae_config.Arch.Predict.output_node)
    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_cvae_config, model_catagoery=model_type)

    if local_test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
        source_data_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/data-sample/'
    else:
        source_data_dir = '/Local-Scratch/oschulte/Galen/2018-2019/'
        data_store_dir = icehockey_cvae_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        # shuffle(dir_games_all)  # randomly shuffle the list
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 8]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: testing
    number_of_total_game = len(dir_games_all)
    icehockey_cvae_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    cvae = CVAE_NN(config=icehockey_cvae_config)
    cvae()
    sess.run(tf.global_variables_initializer())
    if training:
        if not local_test_flag:
            # save the training and testing dir list

            if os.path.exists(saved_network_dir + '/training_file_dirs_all.csv'):
                os.rename(saved_network_dir + '/training_file_dirs_all.csv',
                          saved_network_dir + '/bak_training_file_dirs_all_{0}.csv'
                          .format(datetime.date.today().strftime("%Y%B%d")))
            if os.path.exists(saved_network_dir + '/testing_file_dirs_all.csv'):
                os.rename(saved_network_dir + '/testing_file_dirs_all.csv',
                          saved_network_dir + '/bak_testing_file_dirs_all_{0}.csv'
                          .format(datetime.date.today().strftime("%Y%B%d")))

            if not os.path.exists(saved_network_dir):
                os.mkdir(saved_network_dir)
            with open(saved_network_dir + '/training_file_dirs_all.csv', 'wb') as f:
                for dir in dir_games_all[0: len(dir_games_all) / 10 * 8]:
                    f.write(dir + '\n')
            with open(saved_network_dir + '/testing_file_dirs_all.csv', 'wb') as f:
                for dir in dir_games_all[len(dir_games_all) / 10 * 9:]:
                    f.write(dir + '\n')
        print('training the model.')
        run_network(sess=sess, model=cvae, config=icehockey_cvae_config, log_dir=log_dir,
                    save_network_dir=saved_network_dir, data_store=data_store_dir, source_data_dir=source_data_dir,
                    training_dir_games_all=training_dir_games_all, testing_dir_games_all=testing_dir_games_all,
                    player_id_cluster_dir=player_id_cluster_dir)
        sess.close()
    else:
        print('testing the model')
        model_number = 9301
        testing_dir_games_all = dir_games_all[len(dir_games_all) / 10 * 9:]
        saver = tf.train.Saver()
        model_path = saved_network_dir + '/' + icehockey_cvae_config.Learn.data_name + '-game--{0}'.format(
            str(model_number))
        # save_network_dir + '/' + config.Learn.data_name + '-game-'
        saver.restore(sess, model_path)
        print 'successfully load data from' + model_path

        validate_model(testing_dir_games_all,
                       data_store=data_store_dir,
                       source_data_dir=source_data_dir,
                       config=icehockey_cvae_config,
                       sess=sess,
                       model=cvae,
                       player_id_cluster_dir=player_id_cluster_dir,
                       train_game_number=None,
                       validate_cvrnn_flag=True,
                       validate_td_flag=True,
                       validate_diff_flag=True,
                       validate_predict_flag=True)
        sess.close()


if __name__ == '__main__':
    run()
