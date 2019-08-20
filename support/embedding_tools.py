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
