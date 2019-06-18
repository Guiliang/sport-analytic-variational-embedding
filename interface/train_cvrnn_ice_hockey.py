import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
from config.cvrnn_config import CVRNNCongfig
from nn_structure.cvrnn import CVRNN
from support.data_processing_tools import handle_trace_length, compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv
from support.data_processing_tools import get_icehockey_game_data, transfer2seq
from support.model_tools import get_model_and_log_name


def train_network(sess, model, config, log_dir, saved_network, dir_games_all, data_store, writing_loss_flag=False):
    game_number = 0
    global_counter = 0
    converge_flag = False
    game_diff_record_all = []

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
        for dir_game in dir_games_all:
            if dir_game == '.DS_Store':
                continue
            game_number += 1
            game_cost_record = []
            state_trace_length, state_input, reward, action, team_id = get_icehockey_game_data(
                data_store=data_store, dir_game=dir_game, config=config)
            action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                      max_length=config.Learn.max_seq_length)
            team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                       max_length=config.Learn.max_seq_length)
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

                input_data = np.concatenate([np.asarray(action_id_t0), np.asarray(s_t0_batch),
                                                np.asarray(team_id_t0_batch)], axis=2)
                target_data = np.asarray(action_id_t0)
                trace_length = trace_t0_batch
                [
                    kl_loss,
                    likelihood_loss,
                    _
                ] = sess.run([
                    model.kl_loss,
                    model.likelihood_loss,
                    model.train_op],
                    feed_dict={model.input_data_ph: input_data,
                               model.target_data_ph: target_data,
                               model.trace_length_ph: trace_length}
                )

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][7]
                    # cut = batch_return[i][8]

                # perform gradient step
                # v_diff_record.append(td_loss)

                # if cost_out > 0.0001: # TODO: we still need to consider how to define convergence
                #     converge_flag = False
                global_counter += 1
                cost_out = likelihood_loss + kl_loss
                game_cost_record.append(cost_out)
                # train_writer.add_summary(summary_train, global_step=global_counter)
                s_t0 = s_tl

                print ("cost of the network: kl:{0} and ll:{1}".format(str(np.mean(kl_loss)), str(np.mean(likelihood_loss))))
                # home_avg = sum(q0[:, 0]) / len(q0[:, 0])
                # away_avg = sum(q0[:, 1]) / len(q0[:, 1])
                # end_avg = sum(q0[:, 2]) / len(q0[:, 2])
                # print "home average:{0}, away average:{1}, end average:{2}".format(str(home_avg), str(away_avg),
                #                                                                    str(end_avg))

                if terminal:
                    # save progress after a game
                    model.saver.save(sess, saved_network + '/' + config.learn.sport + '-game-',
                                     global_step=game_number)
                    # v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    # game_diff_record_dict.update({dir_game: v_diff_record_average})
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
    test_flag = False
    icehockey_cvrnn_config_path = "../icehockey_cvrnn_config.yaml"
    icehockey_cvrnn_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)

    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_cvrnn_config)
    if test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
    else:
        data_store_dir = "/cs/oschulte/Galen/Ice-hockey-data/2018-2019/"

    dir_games_all = os.listdir(data_store_dir)
    number_of_total_game = len(dir_games_all)
    icehockey_cvrnn_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    cvrnn = CVRNN(config=icehockey_cvrnn_config)
    cvrnn.call()
    sess.run(tf.global_variables_initializer())
    train_network(sess=sess, model=cvrnn, config=icehockey_cvrnn_config, log_dir=log_dir,
                  saved_network=saved_network_dir, data_store=data_store_dir, dir_games_all=dir_games_all)
    sess.close()


if __name__ == '__main__':
    run()
