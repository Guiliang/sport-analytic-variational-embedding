import matplotlib
matplotlib.use('Agg')
import numpy as np


def plot_players_games(match_q_values_players_dict, iteration):
    matplotlib.pyplot.figure()
    player_ids = match_q_values_players_dict.keys()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)

        matplotlib.pyplot.plot(np.asarray(q_values)[:, 0])
    matplotlib.pyplot.savefig('./test_figures/Q_home_iter{0}.png'.format(str(iteration)))

     matplotlib.pyplot.figure()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)

        matplotlib.pyplot.plot(np.asarray(q_values)[:, 1])
    matplotlib.pyplot.savefig('./test_figures/Q_away_iter{0}.png'.format(str(iteration)))

    matplotlib.pyplot.figure()
    for player_id in player_ids[:3]:
        q_values = match_q_values_players_dict.get(player_id)

        matplotlib.pyplot.plot(np.asarray(q_values)[:, 2])
    matplotlib.pyplot.savefig('./test_figures/Q_end_iter{0}.png'.format(str(iteration)))
    pass
