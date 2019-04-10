import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
from config.cvae_config import CVAECongfig
from nn_structure.cvae_nn import CVAE_NN
from support.data_processing_tools import handle_trace_length, compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv
from support.data_processing_tools import get_icehockey_game_data


def train_network(sess, model, config, log_dir, saved_network, dir_games_all, data_store, print_parameters=False):
    game_number = 0
    global_counter = 0
    converge_flag = False
    game_diff_record_all = []

    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / config.number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= config.number_of_total_game * config.learn.iterate_num:
            break
        else:
            converge_flag = True
        for dir_game in dir_games_all:

            v_diff_record = []
            game_number += 1
            game_cost_record = []
            state_trace_length, state_input, reward, ha_id = get_icehockey_game_data(
                data_store=data_store, dir_game=dir_game, config=config)

            print ("\n load file" + str(dir_game) + " success")
            reward_count = sum(reward)
            print ("reward number" + str(reward_count))
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
                                                         train_number=train_number,
                                                         train_len=train_len,
                                                         state_trace_length=state_trace_length,
                                                         ha_id=ha_id,
                                                         batch_size=config.learn.batch_size)

                # get the batch variables
                s_t0_batch = [d[0] for d in batch_return]
                s_t1_batch = [d[1] for d in batch_return]
                r_t_batch = [d[2] for d in batch_return]
                trace_t0_batch = [d[3] for d in batch_return]
                trace_t1_batch = [d[4] for d in batch_return]
                ha_id_t0_batch = [d[5] for d in batch_return]
                ha_id_t1_batch = [d[6] for d in batch_return]
                y_batch = []

                # readout_t1_batch = model.read_out.eval(
                #     feed_dict={model.trace_lengths: trace_t1_batch, model.rnn_input: s_t1_batch})  # get value of s

                [readout_t1_batch] = sess.run([model.readout],
                                              feed_dict={model.trace_lengths_ph: trace_t1_batch,
                                                         model.rnn_input_ph: s_t1_batch,
                                                         model.home_away_indicator_ph: ha_id_t1_batch
                                                         })

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][7]
                    # cut = batch_return[i][8]

                # perform gradient step
                y_batch = np.asarray(y_batch)
                [diff, read_out, cost_out, summary_train, _] = sess.run(
                    [model.diff, model.readout, model.cost, model.train_step],
                    feed_dict={model.y_ph: y_batch,
                               model.trace_lengths_ph: trace_t0_batch,
                               model.rnn_input_ph: s_t0_batch,
                               model.home_away_indicator_ph: ha_id_t0_batch})

                v_diff_record.append(diff)

                if cost_out > 0.0001:
                    converge_flag = False
                global_counter += 1
                game_cost_record.append(cost_out)
                # train_writer.add_summary(summary_train, global_step=global_counter)
                s_t0 = s_tl

                # print info
                # if print_flag:
                print ("cost of the network is" + str(cost_out))
                # if terminal or ((train_number - 1) / tt_lstm_config.learn.batch_size) % 5 == 1:
                # print ("TIMESTEP:", train_number, "Game:", game_number)
                home_avg = sum(read_out[:, 0]) / len(read_out[:, 0])
                away_avg = sum(read_out[:, 1]) / len(read_out[:, 1])
                end_avg = sum(read_out[:, 2]) / len(read_out[:, 2])
                print "home average:{0}, away average:{1}, end average:{2}".format(str(home_avg), str(away_avg),
                                                                                   str(end_avg))
                # if print_flag:
                #     print ("cost of the network is" + str(cost_out))

                if terminal:
                    # save progress after a game
                    model.saver.save(sess, saved_network + '/' + config.learn.sport + '-game-',
                                     global_step=game_number)
                    v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break

                    # break
            cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
            write_game_average_csv(
                [{"iteration": str(game_number / config.number_of_total_game + 1), "game": game_number,
                  "cost_per_game_average": cost_per_game_average}],
                log_dir=log_dir)

        game_diff_record_all.append(game_diff_record_dict)


def run():
    icehockey_cvae_config_path = "./icehockey_cvae_config.yaml"
    icehockey_cvae_config = CVAECongfig.load(icehockey_cvae_config_path)

    log_dir = icehockey_cvae_config.learn.save_mother_dir + "/oschulte/Galen/icehockey-models/hybrid_sl_log_NN/Scale-tt-three-cut_together_log_train_feature" + str(
        icehockey_cvae_config.learn.feature_type) + "_batch" + str(
        icehockey_cvae_config.learn.batch_size) + "_iterate" + str(
        icehockey_cvae_config.learn.iterate_num) + "_lr" + str(
        icehockey_cvae_config.learn.learning_rate) + "_" + str(
        icehockey_cvae_config.learn.model_type) + icehockey_cvae_config.learn.if_correct_velocity + "_MaxTL" + str(
        icehockey_cvae_config.learn.max_trace_length)
    saved_network_dir = icehockey_cvae_config.learn.save_mother_dir + "/oschulte/Galen/icehockey-models/hybrid_sl_saved_NN/Scale-tt-three-cut_together_saved_networks_feature" + str(
        icehockey_cvae_config.learn.feature_type) + "_batch" + str(
        icehockey_cvae_config.learn.batch_size) + "_iterate" + str(
        icehockey_cvae_config.learn.iterate_num) + "_lr" + str(
        icehockey_cvae_config.learn.learning_rate) + "_" + str(
        icehockey_cvae_config.learn.model_type) + icehockey_cvae_config.learn.if_correct_velocity + "_MaxTL" + str(
        icehockey_cvae_config.learn.max_trace_length)
    data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature" + str(
        icehockey_cvae_config.learn.feature_type) + "-scale-neg_reward" + icehockey_cvae_config.learn.if_correct_velocity + "_length-dynamic"

    dir_games_all = os.listdir(data_store_dir)
    number_of_total_game = len(dir_games_all)
    icehockey_cvae_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    model = CVAE_NN(config=icehockey_cvae_config)
    model.init_placeholder()
    model.build()
    model.call()

    train_network(sess=sess, model=model, config=icehockey_cvae_config, log_dir=log_dir,
                  saved_network=saved_network_dir, data_store=data_store_dir, dir_games_all=dir_games_all)
    sess.close()


if __name__ == '__main__':
    run()
