import json
from player_classification import PlayerCluster
from support.data_processing_tools import read_player_stats

if __name__ == '__main__':
    player_scoring_stats_dir = '../resource/ice_hockey_201819/NHL_player_1819_scoring.csv'
    player_scoring_stats = read_player_stats(player_scoring_stats_dir)
    # check_duplicate_name(player_scoring_stats)

    player_basic_info_dir = '../resource/ice_hockey_201819/player_info_2018_2019.json'
    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info = json.load(f)
    interest_features = ['GP', 'Goals', 'Assists', 'Points', '+/-', 'PIM', 'S']
    cluster_name = 'ap'
    print(cluster_name)
    PC = PlayerCluster(player_basic_info, player_scoring_stats, interest_features, cluster_name)
    PC.gather_cluster_data()
    PC.train_cluster()
    player_id_cluster = PC.obtain_player_cluster()
