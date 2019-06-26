import json
import os
import numpy as np
import pickle

from support.ice_hockey_data_config import player_position_index_dict
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from itertools import cycle
import matplotlib.pyplot as plt


class PlayerCluster(object):
    def __init__(self, soccer_data_dir):
        self.model_save_dir = './cluster_models/ap_cluster_'
        self.soccer_data_dir = soccer_data_dir
        self.position = player_position_index_dict.keys()
        self.position_data_dict = {}
        for action in self.position:
            self.position_data_dict.update({action: {'data': [], 'largest_x': 0, 'largest_y': 0,
                                                     'smallest_x': 100, 'smallest_y': 100}})
        self.loaded_cluster_all = {}

    def load_cluster(self):
        for action in self.position:
            try:
                cluster = pickle.load(open('./generate_cluster/' + self.model_save_dir[2:] + action))
                self.loaded_cluster_all.update({action: cluster})
            except:
                print 'skip cluster {0}'.format(action)
                continue

    def predict_action_cluster(self, events):
        cluster_num_all = []

        for event in events:
            event_action = event['action']
            event_x = event['x']
            event_y = event['y']
            try:
                cluster = self.loaded_cluster_all.get(event_action)
                # print event_action
                cluster_num = cluster.predict([[event_x, event_y]])
                cluster_num_all.append(int(cluster_num[0]))
            except:
                print 'can not find cluster for action {0}'.format(event_action)
                cluster_num_all.append(0)

        return cluster_num_all

    def generate_cluster_data(self, ):
        dir_all = os.listdir(self.soccer_data_dir)
        # self.soccer_data_dir = '/Users/liu/Desktop/'
        # dir_all = ['919069.json']
        for game_dir in dir_all:
            # for i in dir_all[1:11]:

            with open(self.soccer_data_dir + game_dir, 'r') as f:
                game = json.load(f)  # input every game information
            gameId = game['_id']
            events = game['events']

            for event in events:
                event_action = event['action']
                event_x = event['x']
                event_y = event['y']
                action_info = self.position_data_dict.get(event_action)
                action_info.get('data').append([event_x, event_y])
                if event_x > action_info.get('largest_x'):
                    action_info.update({'largest_x': event_x})
                if event_y > action_info.get('largest_y'):
                    action_info.update({'largest_y': event_y})
                if event_x < action_info.get('smallest_x'):
                    action_info.update({'smallest_x': event_x})
                if event_y < action_info.get('smallest_y'):
                    action_info.update({'smallest_y': event_y})
                self.position_data_dict.update({event_action: action_info})

    def train_cluster(self):
        for action in self.position:
            action_info = self.position_data_dict.get(action)
            action_data = action_info.get('data')
            # if action == 'simple-pass':
            #     print 'test'
            if len(action_data) < 2:
                print 'number of cluster for action {0} is 0'.format(action)
                continue
            # preference = -1 * ((action_info.get('largest_x') - action_info.get('smallest_x')) ** 2 +
            #                    (action_info.get('largest_y') - action_info.get('smallest_y')) ** 2) ** 0.5
            # print preference
            # cluster = AffinityPropagation(preference=preference).fit(action_data)
            # cluster = DBSCAN(eps=3, min_samples=2).fit(action_data)
            if len(action_data) <= 6:
                continue
            cluster = KMeans(n_clusters=6).fit(action_data)
            # print cluster.labels_
            cluster_centers_indices = cluster.cluster_centers_
            print 'number of cluster for action {0} is {1}'.format(action, len(cluster_centers_indices))
            with open(self.model_save_dir + action, 'w') as f:
                pickle.dump(cluster, f)
                # self.plot_cluster_results(cluster, action_data)

    def plot_cluster_results(self, cluster, action_data):
        n_clusters_ = len(cluster.cluster_centers_)
        cluster_number_list = cluster.predict(action_data)
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            indices = [i for i, x in enumerate(cluster_number_list) if x == k]
            data_plot = np.asarray([action_data[index] for index in indices])
            plt.scatter(data_plot[:, 0], data_plot[:, 1], c=col)
        plt.show()
