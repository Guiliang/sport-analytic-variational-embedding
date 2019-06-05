import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
from config.cvrnn_config import CVRNNCongfig
from nn_structure.cvrnn import CVRNN
from support.data_processing_tools import handle_trace_length, compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv
from support.data_processing_tools import get_icehockey_game_data
from support.model_tools import get_model_and_log_name


def train_network(sess, model, config, log_dir, saved_network, dir_games_all, data_store, writing_loss_flag=False):
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
            state_trace_length, state_input, reward, ha_id, team_id = get_icehockey_game_data(
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
                                                         team_id=team_id,
                                                         config=config)

                # get the batch variables
                s_t0_batch = [d[0] for d in batch_return]
                s_t1_batch = [d[1] for d in batch_return]
                r_t_batch = [d[2] for d in batch_return]
                # trace_t0_batch = [d[3] for d in batch_return]
                # trace_t1_batch = [d[4] for d in batch_return]
                # ha_id_t0_batch = [d[5] for d in batch_return]
                # ha_id_t1_batch = [d[6] for d in batch_return]
                team_id_t0_batch = [d[7] for d in batch_return]
                team_id_t1_batch = [d[8] for d in batch_return]

                [
                    q0,
                    q1,
                    kl_loss,
                    ll_loss,
                    td_loss,
                    _
                ] = sess.run([
                    model.q_t0_values,
                    model.q_t1_values,
                    model.marginal_likelihood_loss,
                    model.KL_divergence_loss,
                    model.td_loss,
                    model.train_op],
                    feed_dict={model.x_t0_ph: team_id_t0_batch,
                               model.x_t1_ph: team_id_t1_batch,
                               model.y_t0_ph: s_t0_batch,
                               model.y_t1_ph: s_t1_batch,
                               model.reward_ph: r_t_batch}
                )

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][7]
                    # cut = batch_return[i][8]

                # perform gradient step
                v_diff_record.append(td_loss)

                # if cost_out > 0.0001: # TODO: we still need to consider how to define convergence
                #     converge_flag = False
                global_counter += 1
                cost_out = ll_loss + kl_loss + td_loss
                game_cost_record.append(cost_out)
                # train_writer.add_summary(summary_train, global_step=global_counter)
                s_t0 = s_tl

                print ("cost of the network is" + str(cost_out))
                home_avg = sum(q0[:, 0]) / len(q0[:, 0])
                away_avg = sum(q0[:, 1]) / len(q0[:, 1])
                end_avg = sum(q0[:, 2]) / len(q0[:, 2])
                print "home average:{0}, away average:{1}, end average:{2}".format(str(home_avg), str(away_avg),
                                                                                   str(end_avg))

                if terminal:
                    # save progress after a game
                    model.saver.save(sess, saved_network + '/' + config.learn.sport + '-game-',
                                     global_step=game_number)
                    v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break

                    # break
            if writing_loss_flag:
                cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
                write_game_average_csv(
                    [{"iteration": str(game_number / config.number_of_total_game + 1), "game": game_number,
                      "cost_per_game_average": cost_per_game_average}],
                    log_dir=log_dir)

        game_diff_record_all.append(game_diff_record_dict)


def run():
    icehockey_cvrnn_config_path = "./icehockey_cvae_config.yaml"
    icehockey_cvrnn_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)

    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_cvrnn_config)
    data_store_dir = ""

    dir_games_all = os.listdir(data_store_dir)
    number_of_total_game = len(dir_games_all)
    icehockey_cvrnn_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    cvrnn = CVRNN(config=icehockey_cvrnn_config)
    cvrnn.call()

    train_network(sess=sess, model=cvrnn, config=icehockey_cvrnn_config, log_dir=log_dir,
                  saved_network=saved_network_dir, data_store=data_store_dir, dir_games_all=dir_games_all)
    sess.close()


if __name__ == '__main__':
    run()
