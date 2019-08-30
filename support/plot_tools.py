import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_shadow(x_values_list, y_mean_values_list,
                y_lower_values_list, y_upper_values_list,
                sample_size=3):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(sample_size):
        plt.fill_between(x_values_list, y_lower_values_list[:, i],
                         y_upper_values_list[:, i], alpha=.3, color=colors[i], edgecolor="none")
        plt.plot(x_values_list, y_mean_values_list[:, i], linewidth=2, )
    plt.show()


def plot_game_Q_values(Q_values):
    event_numbers = range(0, len(Q_values))
    plt.figure()
    for i in range(0, len(Q_values[0])):
        plt.plot(event_numbers, Q_values[:, i])
    plt.show()


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


if __name__ == '__main__':
    x = np.arange(0.0, 2, 0.01)
    y1 = np.sin(2 * np.pi * x)
    y2 = 1.2 * np.sin(4 * np.pi * x)

    plot_shadow(x, [(y1 + y2) / 2, (np.flip(y1, 0) + np.flip(y2, 0)) / 2], [y1, np.flip(y1, 0)],
                [y2, np.flip(y2, 0)])
