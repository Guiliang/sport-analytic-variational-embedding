import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from support.ice_hockey_data_config import player_position_index_dict
from sklearn.decomposition import PCA


def compute_embedding_average(cluster_dict, player_stats_info, interest_features=[]):
    cluster_sum_dict = {}
    cluster_count_dict = {}
    for id in cluster_dict.keys():
        player_stat_dict = player_stats_info.get(str(id))
        if player_stat_dict is None:
            print(id)
            continue
        feature_list = []
        for feature in interest_features:
            feature_list.append(float(player_stat_dict.get(feature)))
        cluster_id = cluster_dict.get(id)

        if cluster_sum_dict.get(cluster_id):
            cluster_sum_list = cluster_sum_dict.get(cluster_id)
            cluster_sum_list = [x + y for x, y in zip(cluster_sum_list, feature_list)]
            cluster_sum_dict.update({cluster_id: cluster_sum_list})
            count_number = cluster_count_dict.get(cluster_id)
            cluster_count_dict.update({cluster_id: count_number + 1})
        else:
            cluster_sum_dict.update({cluster_id: feature_list})
            cluster_count_dict.update({cluster_id: 1})

    for cluster_id in cluster_sum_dict.keys():
        cluster_sum_list = cluster_sum_dict.get(cluster_id)
        count_number = cluster_count_dict.get(cluster_id)
        average_values = [str(interest_features[i]) + ':' + str(round(cluster_sum_list[i] / count_number, 2))
                          for i in range(len(cluster_sum_list))]
        print('cluster {0}:{1}'.format(str(cluster_id), str(average_values)))


def plot_embeddings(data, cluster_number, if_print=True):
    num_cluster = np.max(cluster_number, axis=0)
    for cluster in range(0, num_cluster + 1):
        indices = [i for i, x in enumerate(cluster_number) if x == cluster]
        plt.scatter(data[indices, 0], data[indices, 1], s=10, label=cluster)

        if if_print:
            x_plot = data[indices, 0].tolist()
            y_plot = data[indices, 1].tolist()
            max_x_index = indices[x_plot.index(max(x_plot))]  # 'find the special player'
            max_y_index = indices[y_plot.index(max(y_plot))]
            min_x_index = indices[x_plot.index(min(x_plot))]  # 'find the special player'
            min_y_index = indices[y_plot.index(min(y_plot))]
            print('cluster {0}, max_x index {1}, max_y index {2}, '
                  'min_x_index {3}, min_y_index{4}'.format(str(cluster), str(max_x_index),
                                                           str(max_y_index), str(min_x_index),
                                                           str(min_y_index)))
    # plt.show()
    plt.legend()
    plt.show()
    # plt.savefig('./player_de_cluster{0}.png'.format(str(num_cluster + 1)))


def get_player_cluster(player_index_list, player_basic_info_dir, clutser_type):
    # if player_basic_info_dir is None:
    #     player_basic_info_dir = '../sport_resource/ice_hockey_201819/player_info_2018_2019.json'
    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info = json.load(f)

    player_index_position_pair = {}
    for player_info in player_basic_info.values():
        index = player_info.get('index')
        position = player_info.get('position')
        player_index_position_pair.update({index: position})

    player_cluster_index_dict = {}
    cluster_id = 0

    player_cluster_list = []
    for player_index in player_index_list:
        if clutser_type == 'position':
            position = player_index_position_pair.get(player_index)
            if player_cluster_index_dict.get(position) is None:
                player_cluster_index_dict.update({position: cluster_id})
                cluster_id += 1
            player_cluster_index = player_cluster_index_dict.get(position)
        elif clutser_type == 'pindex':
            if player_cluster_index_dict.get(player_index) is None:
                player_cluster_index_dict.update({player_index: cluster_id})
                cluster_id += 1
            player_cluster_index = player_cluster_index_dict.get(player_index)
        else:
            raise ValueError('unknown {0}'.format(clutser_type))
        player_cluster_list.append(player_cluster_index)

    print(player_cluster_index_dict)
    return player_cluster_list


def aggregate_positions_within_cluster(player_basic_info, cluster_number):
    player_positions = [0] * len(player_basic_info)
    for player_info in player_basic_info.values():
        index = player_info.get('index')
        position = player_info.get('position')
        player_positions[index] = position

    cluster_position_pairs = zip(cluster_number, player_positions)

    cluster_position_count_dict = {}

    for cluster_position_pair in cluster_position_pairs:
        if cluster_position_count_dict.get(cluster_position_pair[0]):
            cluster_count = cluster_position_count_dict.get(cluster_position_pair[0])
            cluster_count.update({cluster_position_pair[1]: cluster_count[cluster_position_pair[1]] + 1})
            cluster_position_count_dict.update({cluster_position_pair[0]: cluster_count})
        else:
            cluster_count = {'C': 0, 'RW': 0, 'LW': 0, 'D': 0, 'G': 0}
            cluster_count.update({cluster_position_pair[1]: 1})
            cluster_position_count_dict.update({cluster_position_pair[0]: cluster_count})
    print(cluster_position_count_dict)
    for cluster_id in cluster_position_count_dict.keys():
        print('cluster {0} with counts {1}'.format(str(cluster_id), str(cluster_position_count_dict.get(cluster_id))))
    return cluster_position_count_dict


def dimensional_reduction(embeddings, dr_method):
    if dr_method == 'PCA':
        dr_embedding = PCA(n_components=2).fit_transform(embeddings)
        print ('finish pca')
    elif dr_method == 'TSNE':
        dr_embedding = TSNE(n_components=2).fit_transform(embeddings)
        print ('finish t-sne')
    else:
        raise ValueError('unknown {0}'.format(dr_method))
    return dr_embedding
