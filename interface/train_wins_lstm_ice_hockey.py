import sys
import traceback

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
from config.LSTM_win_config import LSTMWinCongfig
from nn_structure.lstm_win_nn import Win_Prediction
from support.data_processing_tools import get_together_training_batch, compute_game_win_vec
from support.data_processing_tools import get_icehockey_game_data, transfer2seq
from support.model_tools import get_model_and_log_name, ExperienceReplayBuffer

MemoryBuffer = ExperienceReplayBuffer(capacity_number=30000)


def gathering_running_and_run(dir_game, config, player_id_cluster_dir, data_store,
                              model, sess, training_flag, game_number, validate_flag=False,
                              output_label_all=None, real_label_all=None
                              ):
    state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
        data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)
    action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                              max_length=config.Learn.max_seq_length)
    team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                               max_length=config.Learn.max_seq_length)
    player_index_seq = transfer2seq(data=player_index, trace_length=state_trace_length,
                                    max_length=config.Learn.max_seq_length)
    win_one_hot = compute_game_win_vec(rewards=reward)
    # reward_count = sum(reward)
    # print ("reward number" + str(reward_count))
    if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
        raise Exception('state length does not equal to reward length')

    train_len = len(state_input)
    train_number = 0
    batch_number = 0
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
                                                 win_info=win_one_hot,
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

        player_embed_t0 = None  # TODO: insert player embedding
        input_data_t0 = np.concatenate([np.asarray(s_t0_batch), np.asarray(action_id_t0)], axis=2)
        trace_lengths_t0 = trace_t0_batch
        outcome_data = win_id_t_batch

        player_embed_t1 = None  # TODO: insert player embedding
        input_data_t1 = np.concatenate([np.asarray(s_t1_batch), np.asarray(action_id_t1)], axis=2)
        trace_lengths_t1 = trace_t1_batch

        for i in range(0, len(batch_return)):
            terminal = batch_return[i][-2]
            cut = batch_return[i][-1]

        if training_flag:
            if config.Learn.apply_stochastic and not config.Learn.apply_rl:
                for i in range(len(input_data_t0)):
                    MemoryBuffer.push([input_data_t0[i], trace_lengths_t0[i],
                                       input_data_t1[i], trace_lengths_t1[i],
                                       r_t_batch[i], win_id_t_batch[i]
                                       ])
                sampled_data = MemoryBuffer.sample(batch_size=config.Learn.batch_size)
                sample_input_data_t0 = np.asarray([sampled_data[j][0] for j in range(len(sampled_data))])
                sample_trace_lengths_t0 = np.asarray([sampled_data[j][1] for j in range(len(sampled_data))])
                sample_input_data_t1 = np.asarray([sampled_data[j][2] for j in range(len(sampled_data))])
                sample_trace_lengths_t1 = np.asarray([sampled_data[j][3] for j in range(len(sampled_data))])
                sampled_outcome_t = np.asarray([sampled_data[j][5] for j in range(len(sampled_data))])
            else:
                sample_input_data_t0 = input_data_t0
                sample_trace_lengths_t0 = trace_lengths_t0
                sample_input_data_t1 = input_data_t1
                sample_trace_lengths_t1 = trace_lengths_t1
                sampled_outcome_t = win_id_t_batch
            print_flag = True if batch_number % (len(state_input) / (config.Learn.batch_size * 10)) == 0 else False
            output_label, real_label = train_win_prob(model, sess, config,
                                                      sample_input_data_t0,
                                                      sample_trace_lengths_t0,
                                                      sample_input_data_t1,
                                                      sample_trace_lengths_t1,
                                                      sampled_outcome_t,
                                                      terminal)
            if real_label_all is None:
                real_label_all = real_label
            else:
                real_label_all = np.concatenate([real_label_all, real_label], axis=0)

            if output_label_all is None:
                output_label_all = output_label
            else:
                output_label_all = np.concatenate([output_label_all, output_label], axis=0)

        if validate_flag:
            output_label, real_label = win_validation(sess, model, input_data_t0,
                                                      trace_lengths_t0,
                                                      config, win_id_t_batch)

            if real_label_all is None:
                real_label_all = real_label
            else:
                real_label_all = np.concatenate([real_label_all, real_label], axis=0)

            if output_label_all is None:
                output_label_all = output_label
            else:
                output_label_all = np.concatenate([output_label_all, output_label], axis=0)
        else:
            pass
        batch_number += 1
        s_t0 = s_tl
        if terminal:
            break

    return [[real_label_all, output_label_all]]


def run_network(sess, model, config, log_dir, save_network_dir,
                training_dir_games_all, testing_dir_games_all,
                data_store, player_id_cluster_dir, save_flag=True):
    game_number = 0
    converge_flag = False
    saver = tf.train.Saver(max_to_keep=300)
    pretrain_number = 500

    real_label_record = np.ones([100, 5000]) * -1
    output_label_record = np.ones([100, 5000]) * -1
    dir_index = 0

    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / config.Learn.number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= len(training_dir_games_all) * config.Learn.iterate_num:
            break
        # else:
        #     converge_flag = False  # TODO: should be set to False?
        for dir_game in training_dir_games_all:
            if dir_game == '.DS_Store':
                continue
            game_number += 1
            print ("\ntraining file {0} with game number {1}".format(str(dir_game), str(game_number)))
            pretrain_flag = True if game_number <= pretrain_number else False

            output_label_all, real_label_all = gathering_running_and_run(dir_game, config,
                                                                         player_id_cluster_dir,
                                                                         data_store, model, sess,
                                                                         training_flag=True,
                                                                         game_number=game_number)

            real_label_record[dir_index][:len(real_label_all)] = real_label_all[:len(real_label_all)]
            output_label_record[dir_index][:len(output_label_all)] = output_label_all[:len(output_label_all)]
            dir_index += 1
            if game_number % 100 == 1:

                for i in range(0, output_label_record.shape[1]):
                    real_outcome_record_step = real_label_record[:, i]
                    model_output_record_step = output_label_record[:, i]

                    correct_num = 0
                    total_number = 0
                    for win_index in range(0, len(real_outcome_record_step)):
                        if model_output_record_step[win_index] == -1 or real_outcome_record_step[win_index] == -1:
                            continue
                        if model_output_record_step[win_index] == real_outcome_record_step[win_index]:
                            correct_num += 1
                        total_number += 1

                    if i % 100 == 0 and total_number > 0:
                        print('acc of time {0} is total:{1}, '
                              'correct:{2}, acc:{3}'.format(str(i),
                                                            str(total_number),
                                                            str(correct_num),
                                                            str(float(correct_num) / total_number)))

                real_label_record = np.ones([100, 5000]) * -1
                output_label_record = np.ones([100, 5000]) * -1
                dir_index = 0

            if game_number % 100 == 1 and save_flag:
                save_model(game_number, saver, sess, save_network_dir, config)
            if game_number % 100 == 1:
                validate_model(testing_dir_games_all, data_store, config,
                               sess, model, player_id_cluster_dir, train_game_number=game_number)


def validate_model(testing_dir_games_all, data_store, config, sess, model,
                   player_id_cluster_dir, train_game_number):
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

        [
            real_label_all,
            output_label_all] = gathering_running_and_run(dir_game, config,
                                                          player_id_cluster_dir,
                                                          data_store,
                                                          model, sess,
                                                          training_flag=False,
                                                          game_number=None,
                                                          validate_flag=True,
                                                          output_label_all=output_label_all,
                                                          real_label_all=real_label_all)

        real_label_record[dir_index][:len(real_label_all)] = real_label_all[:len(real_label_all)]
        output_label_record[dir_index][:len(output_label_all)] = output_label_all[:len(output_label_all)]

    print ('general real label is {0}'.format(str(np.sum(real_label_record, axis=1))))
    print ('general output label is {0}'.format(str(np.sum(output_label_record, axis=1))))
    for i in range(0, output_label_record.shape[1]):
        real_outcome_record_step = real_label_record[:, i]
        model_output_record_step = output_label_record[:, i]

        correct_num = 0
        total_number = 0
        for win_index in range(0, len(real_outcome_record_step)):
            if model_output_record_step[win_index] == -1 or real_outcome_record_step[win_index] == -1:
                continue
            if model_output_record_step[win_index] == real_outcome_record_step[win_index]:
                correct_num += 1
            total_number += 1

        if i % 100 == 0 and total_number > 0:
            print('acc of time {0} is total:{1}, '
                  'correct:{2}, acc:{3}'.format(str(i),
                                                str(total_number),
                                                str(correct_num),
                                                str(float(correct_num) / total_number)))


def save_model(game_number, saver, sess, save_network_dir, config):
    if (game_number - 1) % 300 == 0:
        # save progress after a game
        print 'saving game', game_number
        saver.save(sess, save_network_dir + '/' + config.Learn.sport + '-game-',
                   global_step=game_number)


def train_win_prob(model, sess, config,
                   input_data_t0, trace_lengths_t0,
                   input_data_t1, trace_lengths_t1,
                   outcome_data, terminal):
    if config.Learn.apply_rl:

        [readout_t1_batch] = sess.run([model.read_out],
                                      feed_dict={model.rnn_input_ph: input_data_t1,
                                                 model.trace_lengths_ph: trace_lengths_t1})
        y_batch = []
        for i in range(0, len(readout_t1_batch)):
            # if terminal, only equals reward
            if terminal and i == len(readout_t1_batch) - 1:
                y_home = float(outcome_data[i][0])
                y_away = float(outcome_data[i][1])
                y_end = float(outcome_data[i][2])
                print([y_home, y_away, y_end])
                y_batch.append([y_home, y_away, y_end])
                break
            else:
                y_home = readout_t1_batch[i].tolist()[0]
                y_away = readout_t1_batch[i].tolist()[1]
                y_end = readout_t1_batch[i].tolist()[2]
                y_batch.append([y_home, y_away, y_end])

        train_list = [model.read_out, model.diff, model.train_step]

        train_outputs = \
            sess.run(train_list,
                     feed_dict={model.rnn_input_ph: input_data_t0,
                                model.trace_lengths_ph: trace_lengths_t0,
                                model.y_ph: y_batch
                                })
        print(
            'the avg Q values are home {0}, away {1} and end {2} with diff {3}'.format(np.mean(train_outputs[0][:, 0]),
                                                                                       np.mean(train_outputs[0][:, 1]),
                                                                                       np.mean(train_outputs[0][:, 2]),
                                                                                       np.mean(train_outputs[1])))
        output_label = np.argmax(train_outputs[0], axis=1)
        real_label = np.argmax(outcome_data, axis=1)
    else:

        [
            _,
            win_output
        ] = sess.run([
            model.train_step,
            model.read_out
        ],
            feed_dict={model.rnn_input_ph: input_data_t0,
                       model.y_ph: outcome_data,
                       model.trace_lengths_ph: trace_lengths_t0
                       })

        output_label = np.argmax(win_output, axis=1)
        real_label = np.argmax(outcome_data, axis=1)

        correct_num = 0
        for index in range(0, len(input_data_t0)):
            if output_label[index] == real_label[index]:
                correct_num += 1

    return output_label, real_label


def win_validation(sess, model, input_data,
                   trace_lengths,
                   config, outcome_data):
    [
        # _,
        win_output
    ] = sess.run([
        # model.train_win_op,
        model.read_out],
        feed_dict={model.rnn_input_ph: input_data,
                   model.y_ph: outcome_data,
                   model.trace_lengths_ph: trace_lengths
                   })
    output_label = np.argmax(win_output, axis=1)
    real_label = np.argmax(outcome_data, axis=1)

    # correct_num = 0
    # for index in range(0, len(input_data)):
    #     if output_label[index] == real_label[index]:
    #         correct_num += 1
    #
    return output_label, real_label


def run():
    local_test_flag = False
    training = True
    icehockey_lstm_win_config_path = "../environment_settings/ice_hockey_predict_win_lstm.yaml"
    icehockey_lstm_win_config = LSTMWinCongfig.load(icehockey_lstm_win_config_path)
    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_lstm_win_config, model_catagoery='lstm_win')

    if local_test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
        save_flag = False
    else:
        data_store_dir = icehockey_lstm_win_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 9]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: it is a small testing
        save_flag = True
    number_of_total_game = len(dir_games_all)
    icehockey_lstm_win_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    model = Win_Prediction(config=icehockey_lstm_win_config)
    model()
    sess.run(tf.global_variables_initializer())

    if training:
        if not local_test_flag:
            # save the training and testing dir list
            with open(saved_network_dir + '/training_file_dirs_all.csv') as f:
                for dir in dir_games_all[0: len(dir_games_all) / 10 * 8]:
                    f.write(dir + '\n')
            with open(saved_network_dir + '/testing_file_dirs_all.csv') as f:
                for dir in dir_games_all[len(dir_games_all) / 10 * 9:]:
                    f.write(dir + '\n')
        run_network(sess=sess, model=model, config=icehockey_lstm_win_config, log_dir=log_dir,
                    save_network_dir=saved_network_dir, data_store=data_store_dir,
                    training_dir_games_all=training_dir_games_all, testing_dir_games_all=testing_dir_games_all,
                    player_id_cluster_dir=None, save_flag=save_flag)
        sess.close()
    else:
        model_number = 2101
        testing_dir_games_all = dir_games_all[len(dir_games_all) / 10 * 9:]
        saver = tf.train.Saver()
        model_path = saved_network_dir + '/' + icehockey_lstm_win_config.Learn.sport + '-game--{0}'.format(
            str(model_number))
        saver.restore(sess, model_path)
        print 'successfully load data from' + model_path

        validate_model(testing_dir_games_all,
                       data_store=data_store_dir,
                       config=icehockey_lstm_win_config,
                       sess=sess,
                       model=model,
                       player_id_cluster_dir=None,
                       train_game_number=None)
        sess.close()


if __name__ == '__main__':
    run()
