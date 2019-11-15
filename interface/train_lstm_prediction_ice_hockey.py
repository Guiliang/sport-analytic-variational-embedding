import datetime
import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import os
from support.data_processing_tools import transfer2seq, get_icehockey_game_data, get_together_training_batch, \
    read_feature_within_events
from nn_structure.lstm_prediction_nn import Td_Prediction_NN
from config.lstm_prediction_config import LSTMPredictConfig
from support.model_tools import compute_acc, get_model_and_log_name, BalanceExperienceReplayBuffer, \
    ExperienceReplayBuffer

BalancedMemoryBuffer = BalanceExperienceReplayBuffer(capacity_number=30000)
MemoryBuffer = ExperienceReplayBuffer(capacity_number=30000)


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
    # print ("training acc is {0}".format(str(acc)))


def validation_model(testing_dir_games_all, data_store, config, sess, model, player_id_cluster_dir,
                     source_data_store_dir):
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
            data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)
        action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                  max_length=config.Learn.max_seq_length)
        team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                   max_length=config.Learn.max_seq_length)
        if config.Learn.predict_target == 'ActionGoal' or config.Learn.predict_target == 'Action':
            player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                            max_length=config.Learn.max_seq_length)
        else:
            player_index_seq = player_index
        if config.Learn.predict_target == 'ActionGoal':
            actions_all = read_feature_within_events(directory=dir_game,
                                                     data_path=source_data_store_dir,
                                                     feature_name='name')
            next_goal_label = []
            data_length = state_trace_length.shape[0]

            win_info = []
            new_reward = []
            new_action_seq = []
            new_state_input = []
            new_state_trace_length = []
            new_team_id_seq = []
            for action_index in range(0, data_length):
                action = actions_all[action_index]
                if 'shot' in action:
                    new_reward.append(reward[action_index])
                    new_action_seq.append(action_seq[action_index])
                    new_state_input.append(state_input[action_index])
                    new_state_trace_length.append(state_trace_length[action_index])
                    new_team_id_seq.append(team_id_seq[action_index])
                    if action_index + 1 == data_length:
                        continue
                    if actions_all[action_index + 1] == 'goal':
                        # print(actions_all[action_index+1])
                        next_goal_label.append([1, 0])
                    else:
                        # print(actions_all[action_index + 1])
                        next_goal_label.append([0, 1])
            reward = np.asarray(new_reward)
            action_seq = np.asarray(new_action_seq)
            state_input = np.asarray(new_state_input)
            state_trace_length = np.asarray(new_state_trace_length)
            team_id_seq = np.asarray(new_team_id_seq)
            win_info = np.asarray(next_goal_label)
        elif config.Learn.predict_target == 'Action':
            win_info = action[1:, :]
            reward = reward[:-1]
            action_seq = action_seq[:-1, :, :]
            state_input = state_input[:-1, :, :]
            state_trace_length = state_trace_length[:-1]
            team_id_seq = team_id_seq[:-1, :, :]
        else:
            win_info = None
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
                                                     player_index=player_index_seq,
                                                     train_number=train_number,
                                                     train_len=train_len,
                                                     state_trace_length=state_trace_length,
                                                     action=action_seq,
                                                     team_id=team_id_seq,
                                                     win_info=win_info,
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
            win_info_t0_batch = [d[11] for d in batch_return]

            trace_lengths = trace_t0_batch

            if config.Learn.predict_target == 'ActionGoal':
                target_data = np.asarray(win_info_t0_batch)
            elif config.Learn.predict_target == 'Action':
                target_data = np.asarray(win_info_t0_batch)
            elif config.Learn.predict_target == 'PlayerLocalId':
                config.Learn.apply_pid = False
                target_data = np.asarray(player_id_t0_batch)
            else:
                raise ValueError('unknown predict_target {0}'.format(config.Learn.predict_target))

            if config.Learn.apply_pid:
                input_data = np.concatenate([np.asarray(action_id_t0),
                                             np.asarray(s_t0_batch),
                                             np.asarray(player_id_t0_batch)],
                                            axis=2)
            else:
                input_data = np.concatenate([np.asarray(action_id_t0),
                                             np.asarray(s_t0_batch)],
                                            axis=2)

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

    acc = compute_acc(output_prob=output_decoder_all, target_label=target_data_all, if_print=True)
    print ("testing acc is {0}".format(str(acc)))


def run_network(sess, model, config, training_dir_games_all,
                testing_dir_games_all, model_data_store_dir,
                player_id_cluster_dir, source_data_store_dir, save_network_dir):
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
            game_cost_record = []
            state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
                data_store=model_data_store_dir, dir_game=dir_game, config=config,
                player_id_cluster_dir=player_id_cluster_dir)
            action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                      max_length=config.Learn.max_seq_length)
            team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                       max_length=config.Learn.max_seq_length)
            if config.Learn.predict_target == 'ActionGoal' or config.Learn.predict_target == 'Action':
                player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                                max_length=config.Learn.max_seq_length)
            else:
                player_index_seq = player_index
            if config.Learn.predict_target == 'ActionGoal':
                actions_all = read_feature_within_events(directory=dir_game,
                                                         data_path=source_data_store_dir,
                                                         feature_name='name')
                next_goal_label = []
                data_length = state_trace_length.shape[0]

                win_info = []
                new_reward = []
                new_action_seq = []
                new_state_input = []
                new_state_trace_length = []
                new_team_id_seq = []
                for action_index in range(0, data_length):
                    action = actions_all[action_index]
                    if 'shot' in action:
                        if action_index + 1 == data_length:
                            continue
                        new_reward.append(reward[action_index])
                        new_action_seq.append(action_seq[action_index])
                        new_state_input.append(state_input[action_index])
                        new_state_trace_length.append(state_trace_length[action_index])
                        new_team_id_seq.append(team_id_seq[action_index])
                        if actions_all[action_index + 1] == 'goal':
                            # print(actions_all[action_index+1])
                            next_goal_label.append([1, 0])
                        else:
                            # print(actions_all[action_index + 1])
                            next_goal_label.append([0, 1])
                reward = np.asarray(new_reward)
                action_seq = np.asarray(new_action_seq)
                state_input = np.asarray(new_state_input)
                state_trace_length = np.asarray(new_state_trace_length)
                team_id_seq = np.asarray(new_team_id_seq)
                win_info = np.asarray(next_goal_label)
            elif config.Learn.predict_target == 'Action':
                win_info = action[1:, :]
                reward = reward[:-1]
                action_seq = action_seq[:-1, :, :]
                state_input = state_input[:-1, :, :]
                state_trace_length = state_trace_length[:-1]
                team_id_seq = team_id_seq[:-1, :, :]
            else:
                win_info = None
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
                                                         player_index=player_index_seq,
                                                         train_number=train_number,
                                                         train_len=train_len,
                                                         state_trace_length=state_trace_length,
                                                         action=action_seq,
                                                         team_id=team_id_seq,
                                                         win_info=win_info,
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
                win_info_t0_batch = [d[11] for d in batch_return]

                if config.Learn.predict_target == 'ActionGoal':
                    target_data = np.asarray(win_info_t0_batch)
                    m2balanced = True
                elif config.Learn.predict_target == 'Action':
                    target_data = np.asarray(win_info_t0_batch)
                    m2balanced = True
                elif config.Learn.predict_target == 'PlayerLocalId':
                    config.Learn.apply_pid = False
                    target_data = np.asarray(player_id_t0_batch)
                    m2balanced = False
                else:
                    raise ValueError('unknown predict_target {0}'.format(config.Learn.predict_target))

                if config.Learn.apply_pid:
                    input_data = np.concatenate([np.asarray(action_id_t0),
                                                 np.asarray(s_t0_batch),
                                                 np.asarray(player_id_t0_batch)],
                                                axis=2)
                else:
                    input_data = np.concatenate([np.asarray(action_id_t0),
                                                 np.asarray(s_t0_batch)],
                                                axis=2)
                trace_lengths = trace_t0_batch

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][-2]
                    # cut = batch_return[i][8]

                if config.Learn.apply_stochastic:
                    for i in range(len(input_data)):
                        if m2balanced:
                            cache_label = 0 if target_data[i][0] == 0 else 1
                            BalancedMemoryBuffer.push([input_data[i], target_data[i], trace_lengths[i]],
                                                      cache_label=cache_label)
                        else:
                            MemoryBuffer.push([input_data[i], target_data[i], trace_lengths[i]])
                    if game_number <= 10:
                        s_t0 = s_tl
                        if terminal:
                            break
                        else:
                            continue
                    if m2balanced:
                        sampled_data = BalancedMemoryBuffer.sample(batch_size=config.Learn.batch_size)
                    else:
                        sampled_data = MemoryBuffer.sample(batch_size=config.Learn.batch_size)
                    sample_input_data = np.asarray([sampled_data[j][0] for j in range(len(sampled_data))])
                    sample_target_data = np.asarray([sampled_data[j][1] for j in range(len(sampled_data))])
                    sample_trace_lengths = np.asarray([sampled_data[j][2] for j in range(len(sampled_data))])
                else:
                    sample_input_data = input_data
                    sample_target_data = target_data
                    sample_trace_lengths = trace_lengths

                train_model(model, sess, config, sample_input_data, sample_target_data,
                            sample_trace_lengths, terminal)
                s_t0 = s_tl
                if terminal:
                    # save progress after a game
                    # model.saver.save(sess, saved_network + '/' + config.learn.sport + '-game-',
                    #                  global_step=game_number)
                    # v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    # game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break

            save_model(game_number, saver, sess, save_network_dir, config)

            if game_number % 100 == 1:
                validation_model(testing_dir_games_all, model_data_store_dir, config, sess, model,
                                 player_id_cluster_dir, source_data_store_dir)


def save_model(game_number, saver, sess, save_network_dir, config):
    if (game_number - 1) % 30 == 0:
        # save progress after a game
        print 'saving game', game_number
        saver.save(sess, save_network_dir + '/' + config.Learn.data_name + '-game-',
                   global_step=game_number)


def run():
    play_info = ''
    type = 'pids'
    if type == 'ap_playerId':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/player_id_ap_cluster.json'
        predicted_target = 'PlayerPositionClusterAP'  # playerId_
    elif type == 'km_playerId':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/player_id_km_cluster.json'
        predicted_target = 'PlayerPositionClusterKM'  # playerId_
    elif type == 'pos_playerId':
        player_id_cluster_dir = None
        predicted_target = 'playerposition'
    elif type == 'pids':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/local_player_id_2018_2019.json'
        predicted_target = 'PlayerLocalId'  # playerId_
    elif type == 'action_goal':
        player_id_cluster_dir = None
        predicted_target = 'ActionGoal'
    elif type == 'action':
        player_id_cluster_dir = None
        predicted_target = 'Action'
    else:
        raise ValueError('unknown type')

    BalancedMemoryBuffer.set_cache_memory(cache_number=2)

    tt_lstm_config_path = "../environment_settings/ice_hockey_{0}_prediction{1}.yaml".format(predicted_target,
                                                                                             play_info)
    lstm_prediction_config = LSTMPredictConfig.load(tt_lstm_config_path)

    local_test_flag = False
    saved_network_dir, log_dir = get_model_and_log_name(config=lstm_prediction_config,
                                                        model_catagoery='lstm_prediction')
    if local_test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        source_data_store_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/data-sample/'
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = lstm_prediction_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        source_data_store_dir = lstm_prediction_config.Learn.save_mother_dir + '/oschulte/Galen/2018-2019/'
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 9]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: testing
    number_of_total_game = len(dir_games_all)
    lstm_prediction_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    model = Td_Prediction_NN(
        config=lstm_prediction_config)
    model.initialize_ph()
    model.build()
    model.call()
    sess.run(tf.global_variables_initializer())

    if not local_test_flag:
        if not os.path.exists(saved_network_dir):
            os.mkdir(saved_network_dir)
        # save the training and testing dir list
        if os.path.exists(saved_network_dir + '/training_file_dirs_all.csv'):
            os.rename(saved_network_dir + '/training_file_dirs_all.csv',
                      saved_network_dir + '/bak_training_file_dirs_all_{0}.csv'
                      .format(datetime.date.today().strftime("%Y%B%d")))
        if os.path.exists(saved_network_dir + '/testing_file_dirs_all.csv'):
            os.rename(saved_network_dir + '/testing_file_dirs_all.csv',
                      saved_network_dir + '/bak_testing_file_dirs_all_{0}.csv'
                      .format(datetime.date.today().strftime("%Y%B%d")))
        # save the training and testing dir list
        with open(saved_network_dir + '/training_file_dirs_all.csv', 'wb') as f:
            for dir in dir_games_all[0: len(dir_games_all) / 10 * 8]:
                f.write(dir + '\n')
        with open(saved_network_dir + '/testing_file_dirs_all.csv', 'wb') as f:
            for dir in dir_games_all[len(dir_games_all) / 10 * 9:]:
                f.write(dir + '\n')

    run_network(sess=sess, model=model, config=lstm_prediction_config,
                training_dir_games_all=training_dir_games_all, testing_dir_games_all=testing_dir_games_all,
                model_data_store_dir=data_store_dir, player_id_cluster_dir=player_id_cluster_dir,
                source_data_store_dir=source_data_store_dir, save_network_dir=saved_network_dir)
    sess.close()


if __name__ == '__main__':
    run()
