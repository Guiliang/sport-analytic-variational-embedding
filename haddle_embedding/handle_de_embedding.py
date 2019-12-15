import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.manifold import TSNE
from config.de_embed_config import DEEmbedCongfig
from support.data_processing_tools import read_player_stats, generate_player_name_id_features
from support.embedding_tools import compute_embedding_average, plot_embeddings
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


def merge_embedding_with_player_stats(cluster_number):
    player_basic_info_dir = '../sport_resource/ice_hockey_201819/player_info_2018_2019.json'
    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info = json.load(f)
    player_scoring_stats_dir = '../sport_resource/ice_hockey_201819/NHL_player_1819_scoring.csv'
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
    plot_embeddings(data=dr_embedding, cluster_number=cluster_number)
