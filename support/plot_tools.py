import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection, PatchCollection
from matplotlib import colors as mcolors
import seaborn as sns


# fig = plt.figure()
# ax = fig.gca(projection='3d')


def interpolation_heatmap_example():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata

    # data coordinates and values
    x = np.random.random(100)
    y = np.random.random(100)
    z = np.random.random(100)

    # target grid to interpolate to
    xi = yi = np.arange(0, 1.01, 0.01)
    xi, yi = np.meshgrid(xi, yi)

    # set mask
    # mask = (xi > 0.5) & (xi < 0.6) & (yi > 0.5) & (yi < 0.6)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # mask out the field
    # zi[mask] = np.nan

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi, yi, zi, np.arange(0, 1.01, 0.01))
    plt.plot(x, y, 'k.')
    plt.xlabel('xi', fontsize=16)
    plt.ylabel('yi', fontsize=16)
    plt.show()
    # plt.savefig('interpolated.png', dpi=100)
    # plt.close(fig)


def line_plot_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

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
                sample_size):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(sample_size):
        plt.fill_between(x_values_list, y_lower_values_list[:, i],
                         y_upper_values_list[:, i], alpha=.3, color=colors[i], edgecolor="none")
        plt.plot(x_values_list, y_mean_values_list[:, i], linewidth=2, )
    plt.show()


def plot_diff(game_time_list, diff_values_list, model_category_all):
    # event_numbers = range(0, len(diff_values))
    plt.figure()
    plt.xlabel('Event Number')
    plt.ylabel('Average Difference')
    for i in range(0, len(game_time_list)):
        print('avg of {0} is {1}'.format(model_category_all[i], np.mean(diff_values_list[i])))
        plt.plot(game_time_list[i], diff_values_list[i], label=model_category_all[i])
    plt.legend()
    plt.show()


def plot_game_Q_values(Q_values):
    event_numbers = range(0, len(Q_values))
    plt.figure()

    Q_home = [Q_values[i]['home'] for i in event_numbers]
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


def plot_spatial_projection(value_spatial, save_image_dir=None, save_half_image_dir=None):
    print "heat map"
    plt.figure(figsize=(15, 6))
    sns.set(font_scale=1.6)
    ax = sns.heatmap(value_spatial, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r")

    plt.show()
    if save_image_dir is not None:
        plt.savefig(save_image_dir)

    value_spatial_home_half = [v[200:402] for v in value_spatial]
    plt.figure(figsize=(15, 12))
    sns.set()
    ax = sns.heatmap(value_spatial_home_half, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r")
    plt.show()
    if save_half_image_dir is not None:
        plt.savefig(save_half_image_dir)


if __name__ == '__main__':
    # interpolation_heatmap_example()
    # line_plot_3d()
    x = np.arange(0.0, 2, 0.01)
    y1 = np.sin(2 * np.pi * x)
    y2 = 1.2 * np.sin(4 * np.pi * x)

    plot_shadow(x, np.transpose(np.asarray([(y1 + y2) / 2, (np.flip(y1, 0) + np.flip(y2, 0)) / 2])),
                np.transpose(np.asarray([y1, np.flip(y1, 0)])),
                np.transpose(np.asarray([y2, np.flip(y2, 0)])),
                sample_size=2
                )
