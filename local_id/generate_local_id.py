import json
import os

from support.data_processing_tools import read_player_stats, generate_player_name_id_features


def get_player_by_team(hockey_data_dir):
    data_dir_all = os.listdir(hockey_data_dir)
    players_id_by_team = {}
    for data_dir in data_dir_all:
        print ('processing {0}'.format(str(data_dir)))
        with open(hockey_data_dir + data_dir) as f:
            data = json.load(f)
            rosters = data.get('rosters')
        for teamId in rosters.keys():
            players = rosters.get(teamId)
            if len(players) == 0:
                continue
            player_id_all = set()
            for player_info in players:
                first_name = player_info.get('firstName')
                last_name = player_info.get('lastName')
                position = player_info.get('position')
                id = player_info.get('id')
                player_id_all.add(id)
            if players_id_by_team.get(teamId):
                player_id_all_old = players_id_by_team.get(teamId)
                player_id_all_update = player_id_all if len(player_id_all) > player_id_all_old else player_id_all_old
                players_id_by_team.update({teamId: player_id_all_update})
            else:
                players_id_by_team.update({teamId: player_id_all})
    return players_id_by_team


def get_local_id_by_team(players_id_by_team, player_basic_info_dict, local_id_dir):
    player_scoring_stats_dir = '../resource/ice_hockey_201819/NHL_player_1819_scoring.csv'
    player_scoring_stats = read_player_stats(player_scoring_stats_dir)
    player_id_stats_info_dict = generate_player_name_id_features(player_basic_info_dict,
                                                                 player_scoring_stats,
                                                                 interest_features=['Points'])
    all_local_id_by_team = {}
    for teamId in players_id_by_team.keys():
        player_ids = players_id_by_team.get(teamId)
        player_id_stat_pair = []
        for player_id in player_ids:
            index = player_basic_info_dict.get(player_id).get('index')
            stats_info = player_id_stats_info_dict.get(player_id)
            if stats_info is None:
                points = 0
            else:
                points = stats_info.get('Points')
            player_id_stat_pair.append([index, player_id, points])
        player_id_stat_pair = sorted(player_id_stat_pair, key=lambda x: int(x[2]), reverse=True)
        all_local_id_by_team.update({teamId: player_id_stat_pair})
    with open(local_id_dir, 'w') as f:
        json.dump(all_local_id_by_team, f)


if __name__ == '__main__':
    test_flag = False
    local_id_dir = "../resource/ice_hockey_201819/local_player_id_2018_2019.json"
    if test_flag:
        hockey_data_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/data-sample/'
        save_data_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample'
        player_basic_info_dir = '../resource/ice_hockey_201819/player_info_2018_2019.json'
    else:
        hockey_data_dir = '/cs/oschulte/2019-icehockey-data/2018-2019/'
        save_data_dir = '/Local-Scratch/oschulte/Galen/Ice-hockey-data/2018-2019'
        player_basic_info_dir = '../resource/ice_hockey_201819/player_info_2018_2019.json'
    team_info_dir = '../resource/ice_hockey_201819/teams_NHL.json'

    with open(player_basic_info_dir, 'rb') as f:
        player_basic_info_dict = json.load(f)
    #
    # with open(team_info_dir, 'rb') as f:
    #     team_info_dict = json.load(f)

    players_id_by_team = get_player_by_team(hockey_data_dir)
    get_local_id_by_team(players_id_by_team, player_basic_info_dict, local_id_dir)

    # print('still working')
