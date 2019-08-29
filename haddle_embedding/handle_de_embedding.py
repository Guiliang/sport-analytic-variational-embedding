import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.manifold import TSNE
from config.de_embed_config import DEEmbedCongfig
from support.data_processing_tools import read_player_stats, generate_player_name_id_features
from support.embedding_tools import compute_embedding_average
from support.model_tools import get_model_and_log_name


def combine_de_embeddings():
    combined_embeddings = None
    predicted_target_all = ['state', 'action', 'reward']
    for predicted_target in predicted_target_all:
        de_config_path = "../environment_settings/ice_hockey_{0}_de.yaml".format(predicted_target)
        de_config = DEEmbedCongfig.load(de_config_path)
        save_network_dir, log_dir = get_model_and_log_name(config=de_config, model_catagoery='de_embed',
                                                           train_flag=False)
        save_embed_dir = save_network_dir.replace('de_embed_saved_networks', 'store_embedding'). \
            replace('de_model_saved_NN', 'de_model_save_embedding')

        with open(save_embed_dir + '/embedding_matrix_game{0}.csv'.format(str(901)), 'r') as f:
            csv_reader = csv.reader(f)
            read_embedding = []
            for row in csv_reader:
                # print(row)
                read_embedding.append(row)
        if combined_embeddings is None:
            combined_embeddings = np.asarray(read_embedding)
        else:
            combined_embeddings = np.concatenate([combined_embeddings, read_embedding], axis=1)

    return combined_embeddings


def classify_players(embeddings, classifier_name='ap'):
    if classifier_name == 'km':
        cluster = KMeans(n_clusters=10).fit(embeddings)
        print 'The number of cluster is {0}'.format(len(cluster.cluster_centers_))
    elif classifier_name == 'ap':
        cluster = AffinityPropagation(damping=0.8).fit(embeddings)
        print 'The number of cluster is {0}'.format(len(cluster.cluster_centers_indices_))
    else:
        raise ValueError()
    cluster_number = cluster.predict(embeddings)

    return cluster_number


def dimensional_reduction(embeddings):
    dr_embedding = TSNE(n_components=2).fit_transform(embeddings)
    print 'finish t-sne'
    return dr_embedding


def visualize_embeddings(data, cluster_number, if_print=True):
    num_cluster = np.max(cluster_number, axis=0)
    for cluster in range(0, num_cluster + 1):
        indices = [i for i, x in enumerate(cluster_number) if x == cluster]
        plt.scatter(data[indices, 0], data[indices, 1], s=50, label=cluster)

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
    plt.savefig('./player_de_cluster{0}.png'.format(str(num_cluster + 1)))


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


def merge_embedding_with_player_stats(cluster_number):
    player_basic_info_dir = '../resource/ice_hockey_201819/player_info_2018_2019.json'
    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info = json.load(f)
    player_scoring_stats_dir = '../resource/ice_hockey_201819/NHL_player_1819_scoring.csv'
    player_scoring_stats = read_player_stats(player_scoring_stats_dir)

    cluster_position_count_dict = aggregate_positions_within_cluster(player_basic_info, cluster_number)

    returning_player_id_stats_info_dict = generate_player_name_id_features(player_basic_info, player_scoring_stats,
                                                                           interest_features=['GP', 'Goals', 'Assists',
                                                                                              'Points', '+/-', 'PIM',
                                                                                              'S'])
    player_index_id_map = {}

    for playerId in player_basic_info.keys():
        playerInfo = player_basic_info.get(playerId)
        index = playerInfo.get('index')
        player_index_id_map.update({index: playerId})

    cluster_dict = {}
    for index in range(0, len(cluster_number)):
        cluster_dict.update({player_index_id_map[index]: cluster_number[index]})

    compute_embedding_average(cluster_dict, returning_player_id_stats_info_dict,
                              interest_features=['GP', 'Goals', 'Assists',
                                                 'Points', '+/-', 'PIM', 'S'])


if __name__ == '__main__':
    combined_embeddings = combine_de_embeddings()
    dr_embedding = dimensional_reduction(embeddings=combined_embeddings)
    cluster_number = classify_players(embeddings=combined_embeddings, classifier_name='km')
    merge_embedding_with_player_stats(cluster_number)
    # visualize_embeddings(data=dr_embedding, cluster_number=cluster_number)
