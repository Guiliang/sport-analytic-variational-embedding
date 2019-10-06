import json
import os
import numpy as np
import pickle

from support.ice_hockey_data_config import player_position_index_dict
from support.data_processing_tools import generate_player_name_id_features
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from itertools import cycle
import matplotlib.pyplot as plt


class PlayerCluster(object):
    def __init__(self, player_basic_info, player_scoring_stats, interest_features, cluster_name):
        self.interest_features = interest_features
        self.cluster_name = cluster_name
        self.model_save_dir = './cluster_models/{0}_cluster_'.format(cluster_name)
        self.positions = player_position_index_dict.keys()
        self.position_data_dict = {}
        for position in self.positions:
            self.position_data_dict.update({position: {'data': [], 'id': []}})
        self.loaded_cluster_all = {}
        self.player_basic_info = player_basic_info
        self.player_scoring_stats = player_scoring_stats

    def load_cluster(self):
        for position in self.positions:
            try:
                cluster = pickle.load(open(self.model_save_dir + position))
                self.loaded_cluster_all.update({position: cluster})
            except:
                print 'skip cluster {0}'.format(position)
                continue

    # def predict_action_cluster(self, events):
    #     cluster_num_all = []
    #
    #     for event in events:
    #         event_action = event['action']
    #         event_x = event['x']
    #         event_y = event['y']
    #         try:
    #             cluster = self.loaded_cluster_all.get(event_action)
    #             # print event_action
    #             cluster_num = cluster.predict([[event_x, event_y]])
    #             cluster_num_all.append(int(cluster_num[0]))
    #         except:
    #             print 'can not find cluster for action {0}'.format(event_action)
    #             cluster_num_all.append(0)
    #
    #     return cluster_num_all

    def obtain_player_cluster(self):
        self.load_cluster()
        base_number = 0
        player_id_cluster = {}
        for position in self.positions:
            cluster = self.loaded_cluster_all.get(position)
            position_info = self.position_data_dict.get(position)
            position_data = position_info.get('data')
            position_ids = position_info.get('id')
            cluster_numbers = cluster.predict(position_data)

            for index in range(0, len(position_data)):
                player_id_cluster.update({position_ids[index]: base_number+cluster_numbers[index]})

            cluster_set = set()
            for cluster_number in cluster_numbers:
                cluster_set.add(cluster_number)
            base_number += len(cluster_set)

        with open('../sport_resource/ice_hockey_201819/player_id_{0}_cluster.json'.format(self.cluster_name), 'w') as f:
            json.dump(player_id_cluster, f)

        return player_id_cluster

    def gather_cluster_data(self, ):

        returning_player_id_stats_info_dict = generate_player_name_id_features(self.player_basic_info,
                                                                               self.player_scoring_stats,
                                                                               interest_features=self.interest_features)

        for player_id in returning_player_id_stats_info_dict.keys():
            player_info = returning_player_id_stats_info_dict.get(player_id)
            player_position = player_info.get('position')

            position_training_data = self.position_data_dict.get(player_position).get('data')
            position_training_id = self.position_data_dict.get(player_position).get('id')
            position_training_id.append(player_id)
            position_training_player_data = []
            for feature in self.interest_features:
                position_training_player_data.append(float(player_info.get(feature)))
            position_training_data.append(position_training_player_data)
            self.position_data_dict.update(
                {player_position: {'data': position_training_data, 'id': position_training_id}})

    def train_cluster(self):
        for position in self.positions:
            position_info = self.position_data_dict.get(position)
            position_data = position_info.get('data')
            # if action == 'simple-pass':
            #     print 'test'
            if len(position_data) < 2:
                print 'number of cluster for action {0} is 0'.format(position)
                continue
            # preference = -1 * ((action_info.get('largest_x') - action_info.get('smallest_x')) ** 2 +
            #                    (action_info.get('largest_y') - action_info.get('smallest_y')) ** 2) ** 0.5
            # print preference

            # cluster = DBSCAN(eps=3, min_samples=2).fit(action_data)
            if self.cluster_name == 'km':
                cluster = KMeans(n_clusters=5).fit(position_data)
                print 'number of cluster for action {0} is {1}'.format(position, len(cluster.cluster_centers_))
            elif self.cluster_name == 'ap':
                cluster = AffinityPropagation(damping=0.9).fit(position_data)
                print 'number of cluster for action {0} is {1}'.format(position, len(cluster.cluster_centers_indices_))

            with open(self.model_save_dir + position, 'w') as f:
                pickle.dump(cluster, f)
                # self.plot_cluster_results(cluster, position_data)

    def plot_cluster_results(self, cluster, position_data):
        n_clusters_ = len(cluster.cluster_centers_)
        cluster_number_list = cluster.predict(position_data)
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            indices = [i for i, x in enumerate(cluster_number_list) if x == k]
            data_plot = np.asarray([position_data[index] for index in indices])
            plt.scatter(data_plot[:, 0], data_plot[:, 1], c=col)
        plt.show()
