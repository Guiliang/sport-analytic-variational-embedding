import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_players_games(match_q_values_players_dict, iteration):
    plt.figure()
    player_ids = match_q_values_players_dict.keys()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)
        plt.plot(np.asarray(q_values)[:, 0])
        plt.savefig('./test_figures/Q_home_iter{0}.png'.format(str(iteration)))

    plt.figure()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)

        plt.plot(np.asarray(q_values)[:, 1])
    plt.savefig('./test_figures/Q_away_iter{0}.png'.format(str(iteration)))

    plt.figure()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)

        plt.plot(np.asarray(q_values)[:, 2])
    plt.savefig('./test_figures/Q_end_iter{0}.png'.format(str(iteration)))
    pass
