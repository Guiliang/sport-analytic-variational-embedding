import sys
import traceback

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
from support.data_processing_tools import get_icehockey_game_data, transfer2seq, generate_selection_matrix
from support.model_tools import get_model_and_log_name, compute_rnn_acc


def run_network(sess, model, config, log_dir, saved_network,
                training_dir_games_all, testing_dir_games_all,
                data_store, player_id_cluster_dir):
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
                data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)
            action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                      max_length=config.Learn.max_seq_length)
            team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
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
                train_flag = np.asarray([[[1]]*config.Learn.max_seq_length]*len(s_t0_batch))
                input_data = np.concatenate([np.asarray(action_id_t0), np.asarray(s_t0_batch),
                                             np.asarray(team_id_t0_batch), train_flag], axis=2)
                target_data = np.asarray(action_id_t0)
                trace_lengths = trace_t0_batch
                selection_matrix = generate_selection_matrix(trace_lengths,
                                                             max_trace_length=config.Learn.max_seq_length)

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][-2]
                    # cut = batch_return[i][8]

                pretrain_flag = True if game_number <= 10 else False
                train_model(model, sess, config, input_data, target_data,
                            trace_lengths, selection_matrix, terminal, pretrain_flag)
                s_t0 = s_tl
                if terminal:
                    # save progress after a game
                    # model.saver.save(sess, saved_network + '/' + config.learn.sport + '-game-',
                    #                  global_step=game_number)
                    # v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    # game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break
            if game_number % 10 == 1:
                validation_model(testing_dir_games_all, data_store, config, sess, model, player_id_cluster_dir)


def train_model(model, sess, config, input_data, target_data, trace_lengths, selection_matrix, terminal,
                pretrain_flag=False):
    if pretrain_flag:
        [
            output_x,
            # total_loss,
            kl_loss,
            likelihood_loss,
            _
        ] = sess.run([
            model.output,
            # model.cost,
            model.kl_loss,
            model.likelihood_loss,
            model.train_ll_op],
            feed_dict={model.input_data_ph: input_data,
                       model.target_data_ph: target_data,
                       model.trace_length_ph: trace_lengths,
                       model.selection_matrix_ph: selection_matrix}
        )
    else:
        [
            output_x,
            # total_loss,
            kl_loss,
            likelihood_loss,
            _
        ] = sess.run([
            model.output,
            # model.cost,
            model.kl_loss,
            model.likelihood_loss,
            model.train_general_op],
            feed_dict={model.input_data_ph: input_data,
                       model.target_data_ph: target_data,
                       model.trace_length_ph: trace_lengths,
                       model.selection_matrix_ph: selection_matrix}
        )
    output_decoder = []
    for batch_index in range(0, len(output_x)):
        output_decoder_batch = []
        for trace_length_index in range(0, config.Learn.max_seq_length):
            if selection_matrix[batch_index][trace_length_index]:
                output_decoder_batch.append(output_x[batch_index][trace_length_index])
            else:
                output_decoder_batch.append(np.asarray([0] * config.Learn.action_number))
        output_decoder.append(output_decoder_batch)
    output_decoder = np.asarray(output_decoder)

    acc = compute_rnn_acc(output_prob=output_decoder, target_label=target_data,
                          selection_matrix=selection_matrix, config=config)

    # perform gradient step
    # v_diff_record.append(td_loss)

    # if cost_out > 0.0001: # TODO: we still need to consider how to define convergence
    #     converge_flag = False
    cost_out = likelihood_loss + kl_loss
    # game_cost_record.append(cost_out)
    # train_writer.add_summary(summary_train, global_step=global_counter)
    # if pretrain_flag:
    #     print("cost of the network: kl:0.0 and ll:{1} with acc {2}".format(str(np.mean(kl_loss)),
    #                                                                        str(np.mean(likelihood_loss)),
    #                                                                        str(acc)))
    # else:
    #     print ("cost of the network: kl:{0} and ll:{1} with acc {2}".format(str(np.mean(kl_loss)),
    #                                                                         str(np.mean(likelihood_loss)),
    #                                                                         str(acc)))
    # home_avg = sum(q0[:, 0]) / len(q0[:, 0])
    # away_avg = sum(q0[:, 1]) / len(q0[:, 1])
    # end_avg = sum(q0[:, 2]) / len(q0[:, 2])
    # print "home average:{0}, away average:{1}, end average:{2}".format(str(home_avg), str(away_avg),
    #                                                                    str(end_avg))
    # if writing_loss_flag:
    #     cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
    #     write_game_average_csv(
    #         [{"iteration": str(game_number / config.number_of_total_game + 1), "game": game_number,
    #           "cost_per_game_average": cost_per_game_average}],
    #         log_dir=log_dir)


def validation_model(testing_dir_games_all, data_store, config, sess, model, player_id_cluster_dir):
    output_decoder_all = None
    target_data_all = None
    selection_matrix_all = None
    print('validating model')
    for dir_game in testing_dir_games_all:

        if dir_game == '.DS_Store':
            continue
        try:
            state_trace_length, state_input, reward, action, team_id, player_index = get_icehockey_game_data(
                data_store=data_store, dir_game=dir_game, config=config, player_id_cluster_dir=player_id_cluster_dir)
            action_seq = transfer2seq(data=action, trace_length=state_trace_length,
                                      max_length=config.Learn.max_seq_length)
            team_id_seq = transfer2seq(data=team_id, trace_length=state_trace_length,
                                       max_length=config.Learn.max_seq_length)
        except:
            print(dir_game)
            traceback.print_exc(file=sys.stdout)
        # print ("\n load file" + str(dir_game) + " success")
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
            train_flag = np.asarray([[[0]] * config.Learn.max_seq_length] * len(s_t0_batch))
            input_data = np.concatenate([np.asarray(action_id_t0), np.asarray(s_t0_batch),
                                         np.asarray(team_id_t0_batch), train_flag], axis=2)
            target_data = np.asarray(action_id_t0)

            trace_lengths = trace_t0_batch

            selection_matrix = generate_selection_matrix(trace_lengths,
                                                         max_trace_length=config.Learn.max_seq_length)

            for i in range(0, len(batch_return)):
                terminal = batch_return[i][-2]

            [
                output_x,
            ] = sess.run([
                model.output],
                feed_dict={model.input_data_ph: input_data,
                           model.target_data_ph: target_data,
                           model.trace_length_ph: trace_lengths,
                           model.selection_matrix_ph: selection_matrix}
            )

            output_decoder = []
            for batch_index in range(0, len(output_x)):
                output_decoder_batch = []
                for trace_length_index in range(0, config.Learn.max_seq_length):
                    if selection_matrix[batch_index][trace_length_index]:
                        output_decoder_batch.append(output_x[batch_index][trace_length_index])
                    else:
                        output_decoder_batch.append(np.asarray([0] * config.Learn.action_number))
                output_decoder.append(output_decoder_batch)
            output_decoder = np.asarray(output_decoder)

            if output_decoder_all is None:
                output_decoder_all = output_decoder
                target_data_all = target_data
                selection_matrix_all = selection_matrix
            else:
                output_decoder_all = np.concatenate([output_decoder_all, output_decoder], axis=0)
                target_data_all = np.concatenate([target_data_all, target_data], axis=0)
                selection_matrix_all = np.concatenate([selection_matrix_all, selection_matrix], axis=0)

            s_t0 = s_tl
            if terminal:
                break

    acc = compute_rnn_acc(output_prob=output_decoder_all, target_label=target_data_all,
                          selection_matrix=selection_matrix_all, config=config, if_print=True)
    print ("testing acc is {0}".format(str(acc)))


def run():
    cluster='km'
    if cluster == 'ap':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/player_id_ap_cluster.json'
        predicted_target = '_PlayerPositionClusterAP'  # playerId_
    elif cluster == 'km':
        player_id_cluster_dir = '../sport_resource/ice_hockey_201819/player_id_km_cluster.json'
        predicted_target = '_PlayerPositionClusterKM'  # playerId_
    else:
        player_id_cluster_dir = None
        predicted_target = ''

    test_flag = True
    icehockey_cvrnn_config_path = "../icehockey_cvrnn{0}_config.yaml".format(predicted_target)
    icehockey_cvrnn_config = CVRNNCongfig.load(icehockey_cvrnn_config_path)

    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_cvrnn_config)
    if test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
    else:
        data_store_dir = "/cs/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 9]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: testing
    number_of_total_game = len(dir_games_all)
    icehockey_cvrnn_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    cvrnn = CVRNN(config=icehockey_cvrnn_config)
    cvrnn.call
    sess.run(tf.global_variables_initializer())
    run_network(sess=sess, model=cvrnn, config=icehockey_cvrnn_config, log_dir=log_dir,
                saved_network=saved_network_dir, data_store=data_store_dir,
                training_dir_games_all=training_dir_games_all, testing_dir_games_all=testing_dir_games_all,
                player_id_cluster_dir=player_id_cluster_dir)
    sess.close()


if __name__ == '__main__':
    run()
