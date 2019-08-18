import sys
import traceback

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
from config.mdn_Qs_config import MDNQsCongfig
from nn_structure.mdn_nn import MixtureDensityNN
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
            # print (len(state_input) / (config.Learn.batch_size*10))
            print_flag = True if batch_number % (len(state_input) / (config.Learn.batch_size*10)) == 0 else False
            pretrain_flag = True if game_number < 500 else False
            train_td_model(model, sess, config, input_data_t0, trace_lengths_t0, player_embed_t0,
                           input_data_t1, trace_lengths_t1, player_embed_t1, r_t_batch, terminal, cut,
                           pretrain_flag, print_flag)
        else:
            pass

        batch_number += 1
        s_t0 = s_tl
        if terminal:
            break

    return [output_decoder_all, target_data_all, selection_matrix_all, q_values_all, match_q_values_players_dict]


def run_network(sess, model, config, log_dir, save_network_dir,
                training_dir_games_all, testing_dir_games_all,
                data_store, player_id_cluster_dir, save_flag=True):
    game_number = 0
    converge_flag = False
    saver = tf.train.Saver(max_to_keep=300)

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
            print ("\ntraining file" + str(dir_game))
            gathering_running_and_run(dir_game, config,
                                      player_id_cluster_dir, data_store, model, sess,
                                      training_flag=True, game_number=game_number)
            if game_number % 100 == 1 and save_flag:
                save_model(game_number, saver, sess, save_network_dir, config)
                # validate_model(testing_dir_games_all, data_store, config,
                #                sess, model, player_id_cluster_dir, train_game_number=game_number)


def save_model(game_number, saver, sess, save_network_dir, config):
    if (game_number - 1) % 300 == 0:
        # save progress after a game
        print 'saving game', game_number
        saver.save(sess, save_network_dir + '/' + config.Learn.sport + '-game-',
                   global_step=game_number)


def train_td_model(model, sess, config, input_data_t0, trace_lengths_t0, player_embed_t0,
                   input_data_t1, trace_lengths_t1, player_embed_t1, r_t_batch, terminal, cut,
                   pretrain_flag, print_flag):
    [mu1, var1] = sess.run([model.mu_out, model.var_out],
                           feed_dict={model.rnn_input_ph: input_data_t1,
                                      model.trace_lengths_ph: trace_lengths_t1})
    # for reward in r_t_batch:
    #     if reward[0] == 1 or reward[1] == 1 or reward[2] == 1:
    #         print reward

    if pretrain_flag:
        train_list = [model.mu_out, model.var_out, model.train_pretrain_step, model.td_loss, model.bias_loss]
    else:
        train_list = [model.mu_out, model.var_out, model.train_normal_step, model.normal_loss]

    train_outputs = \
        sess.run(
            train_list,
            feed_dict={model.rnn_input_ph: input_data_t0,
                       model.trace_lengths_ph: trace_lengths_t0,
                       model.y_mu_ph: mu1,
                       model.y_var_ph: var1,
                       model.r_ph: r_t_batch
                       }
        )
    # if normal_diff > 1:
    #     print (mu0)
    #     print (mu1)
    #     print (var0)
    #     print (var1)
    #     print (r_t_batch)
    #     print("loss:{0} gaussian diff:{1}".format(str(loss), str(normal_diff)))
    #     raise ValueError('loss error')
    if print_flag or terminal:
        print(np.mean(mu1, axis=0))
        print(np.mean(var1, axis=0))
        if pretrain_flag:
            [mu0, var0, _, td_loss, bias_loss] = train_outputs
            print("pre train td loss:{0} and bias loss:{1}".format(str(td_loss), str(bias_loss)))
        else:
            [mu0, var0, _, loss] = train_outputs
            print("mdn train loss:{0}".format(str(loss)))


def run():
    test_flag = False
    icehockey_mdn_Qs_config_path = "../environment_settings/ice_hockey_predict_Qs_mdn.yaml"
    icehockey_mdn_Qs_config = MDNQsCongfig.load(icehockey_mdn_Qs_config_path)
    saved_network_dir, log_dir = get_model_and_log_name(config=icehockey_mdn_Qs_config, model_catagoery='mdn_Qs')

    if test_flag:
        data_store_dir = "/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = os.listdir(data_store_dir)
        testing_dir_games_all = os.listdir(data_store_dir)
        save_flag = False
    else:
        data_store_dir = icehockey_mdn_Qs_config.Learn.save_mother_dir + "/oschulte/Galen/Ice-hockey-data/2018-2019/"
        dir_games_all = os.listdir(data_store_dir)
        training_dir_games_all = dir_games_all[0: len(dir_games_all) / 10 * 9]
        # testing_dir_games_all = dir_games_all[len(dir_games_all)/10*9:]
        testing_dir_games_all = dir_games_all[-10:]  # TODO: testing
        save_flag = True
    number_of_total_game = len(dir_games_all)
    icehockey_mdn_Qs_config.Learn.number_of_total_game = number_of_total_game

    sess = tf.Session()
    model = MixtureDensityNN(config=icehockey_mdn_Qs_config)
    model()
    sess.run(tf.global_variables_initializer())
    run_network(sess=sess, model=model, config=icehockey_mdn_Qs_config, log_dir=log_dir,
                save_network_dir=saved_network_dir, data_store=data_store_dir,
                training_dir_games_all=training_dir_games_all, testing_dir_games_all=testing_dir_games_all,
                player_id_cluster_dir=None, save_flag=save_flag)
    sess.close()


if __name__ == '__main__':
    run()
