import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection, PatchCollection
from matplotlib import colors as mcolors

fig = plt.figure()
ax = fig.gca(projection='3d')


def line_plot_3d():
    def cc(arg):
        return mcolors.to_rgba(arg, alpha=0.6)

    xs = np.arange(0, 10, 0.4)
    verts = []
    zs = [0.0, 2.0, 4.0, 6.0]
    for z in zs:
        ys = np.random.rand(len(xs))
        # ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))

    # poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'), cc('y')])
    poly = LineCollection(verts, colors=[cc('r'), cc('g'), cc('b'), cc('y')])
    poly.set_alpha(1)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 7)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 1)

    plt.show()


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

    Q_home = [Q_values[i]['home']for i in event_numbers]
    Q_away = [Q_values[i]['away'] for i in event_numbers]
    Q_end = [Q_values[i]['end'] for i in event_numbers]
    plt.plot(event_numbers, Q_home, label='home')
    plt.plot(event_numbers, Q_away, label='away')
    plt.plot(event_numbers, Q_end, label='end')

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
    line_plot_3d()
    # x = np.arange(0.0, 2, 0.01)
    # y1 = np.sin(2 * np.pi * x)
    # y2 = 1.2 * np.sin(4 * np.pi * x)
    #
    # plot_shadow(x, [(y1 + y2) / 2, (np.flip(y1, 0) + np.flip(y2, 0)) / 2], [y1, np.flip(y1, 0)],
    #             [y2, np.flip(y2, 0)])
