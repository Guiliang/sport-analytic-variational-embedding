import csv
import json

from sklearn import preprocessing

from support.data_processing_tools import match_player_name_NHLcom


def collect_player_stats_by_date(player_data_stats_dir):
    item_names = None
    player_stats_gbg_all = {}
    with open(player_data_stats_dir, 'r') as csvfile:
        data_all = csv.reader(csvfile)
        skip_flag = True
        for data_row in data_all:
            if skip_flag:
                item_names = data_row
                skip_flag = False
                continue
            else:
                player_name = data_row[0]
                home_team = data_row[1]
                away_team = data_row[3].split('vs')[1].strip()
                game_date = int(data_row[3].split('vs')[0].strip().replace('/', ''))

                data_gbg_stats_list = player_stats_gbg_all.get(player_name)
                if data_gbg_stats_list is None:
                    data_gbg_stats_list = []
                data_gbg_stats_item = {'game_date': game_date, 'home_team': home_team, 'away_team': away_team}
                for item_number in range(4, len(data_row)):
                    data_gbg_stats_item.update({item_names[item_number]: float(data_row[item_number])})
                data_gbg_stats_list.append(data_gbg_stats_item)
                player_stats_gbg_all[player_name] = data_gbg_stats_list
    item_names_return = []
    for item_number in range(0, len(item_names)):
        if '%' in item_names[item_number] or '/GP' in item_names[item_number]:
            continue
        item_names_return.append(item_names[item_number])
    return player_stats_gbg_all, item_names_return


def horizontal_rescale(player_stats_summary, item_names, match_name_info_dict):
    scaled_player_stats_summary = {}
    max_game_num = 0
    player_all = player_stats_summary.keys()
    for player_name in player_all:
        data_gbg_summary_list = player_stats_summary.get(player_name)
        max_game_num = len(data_gbg_summary_list) if max_game_num < len(data_gbg_summary_list) else max_game_num
        scaled_player_stats_summary.update({player_name: []})

    for round_index in range(0, max_game_num):
        data_round_all = []
        for player_name in player_all:
            data_gbg_summary_list = player_stats_summary.get(player_name)
            if len(data_gbg_summary_list) > round_index:
                select_round_index = round_index
            else:
                select_round_index = len(data_gbg_summary_list) - 1
            player_item_data_list = []
            for item_number in range(4, len(item_names)):
                player_item_data_list.append(data_gbg_summary_list[select_round_index][item_names[item_number]])
            data_round_all.append(player_item_data_list)
        scaler = preprocessing.StandardScaler().fit(data_round_all)
        data_scale_round_all = scaler.transform(data_round_all)
        # print(data_scale_round_all[0][20 - 4])
        for player_index in range(0, len(player_all)):
            player_name = player_all[player_index]
            data_gbg_summary_list = player_stats_summary.get(player_name)
            if len(data_gbg_summary_list) > round_index:
                scaled_data_gbg_summary_list = scaled_player_stats_summary[player_name]
                scaled_data_gbg_summary_item = {'game_date': data_gbg_summary_list[round_index]['game_date'],
                                                'home_team': data_gbg_summary_list[round_index]['home_team'],
                                                'away_team': data_gbg_summary_list[round_index]['away_team']}

                for item_number in range(4, len(item_names)):
                    # print(item_number)
                    scaled_data_gbg_summary_item.update(
                        {item_names[item_number]: data_scale_round_all[player_index][item_number - 4]})
                    # print(data_scale_round_all[player_index][item_number - 4])
                scaled_data_gbg_summary_list.append(scaled_data_gbg_summary_item)
                scaled_player_stats_summary[player_name] = scaled_data_gbg_summary_list

    scaled_playerIndex_stats_summary = {}
    for player_name in scaled_player_stats_summary.keys():
        # "{'id': match_id, 'position': player_position_basic, 'index': player_index_basic}"
        scale_data_gbg_summary_list = scaled_player_stats_summary.get(player_name)
        player_info = match_name_info_dict.get(player_name)
        if player_info is None:
            print(player_name)
            continue
        scaled_playerIndex_stats_summary.update(
            {player_info['index']: {'player_name': player_name, 'id': player_info['id'],
                                    'position': player_info['position'],
                                    'gbg_summary_list': scale_data_gbg_summary_list}})

    with open("../resource/ice_hockey_201819/" + 'Scale_H_NHL_players_game_summary_201819.csv', 'w') as outfile:
        json.dump(scaled_playerIndex_stats_summary, outfile)
    return scaled_player_stats_summary


def common_rescale(player_stats_summary, item_names):
    scaled_player_stats_summary = {}
    data_all = []
    player_all = player_stats_summary.keys()
    for player_name in player_all:
        data_gbg_summary_list = player_stats_summary.get(player_name)
        # scaled_player_stats_summary.update({player_name: []})
        for data_gbg_summary_item in data_gbg_summary_list:
            player_item_data_list = []
            for item_number in range(4, len(item_names)):
                player_item_data_list.append(data_gbg_summary_item[item_names[item_number]])
            data_all.append(player_item_data_list)

    scaler = preprocessing.StandardScaler().fit(data_all)
    data_scale_round_all = scaler.transform(data_all)
    data_index = 0
    for player_name in player_all:
        data_gbg_summary_list = player_stats_summary.get(player_name)
        scaled_data_gbg_summary_list = []
        for data_gbg_summary_item in data_gbg_summary_list:
            scaled_data_gbg_summary_item = {'game_date': data_gbg_summary_item['game_date'],
                                            'home_team': data_gbg_summary_item['home_team'],
                                            'away_team': data_gbg_summary_item['away_team']}

            for item_number in range(4, len(item_names)):
                # print(item_number)
                scaled_data_gbg_summary_item.update(
                    {item_names[item_number]: data_scale_round_all[data_index][item_number - 4]})
                # print(data_scale_round_all[player_index][item_number - 4])
            scaled_data_gbg_summary_list.append(scaled_data_gbg_summary_item)
            data_index += 1
        scaled_player_stats_summary[player_name] = scaled_data_gbg_summary_list

    with open("../resource/ice_hockey_201819/" + 'Scale_C_NHL_players_game_summary_201819.csv', 'w') as outfile:
        json.dump(scaled_player_stats_summary, outfile)
    return scaled_player_stats_summary


def aggregate_player_stats_by_date(player_stats_gbg_all, item_names, match_name_info_dict):
    player_stats_summary = {}
    for player_name in player_stats_gbg_all.keys():
        data_gbg_stats_list = player_stats_gbg_all.get(player_name)
        data_gbg_stats_list = sorted(data_gbg_stats_list, key=lambda pair: pair['game_date'])
        init_data_gbg_summary_base = {'game_date': None, 'home_team': None, 'away_team': None}
        for item_number in range(4, len(item_names)):
            init_data_gbg_summary_base[item_names[item_number]] = 0
        data_gbg_summary_list = []
        for i in range(0, len(data_gbg_stats_list)):
            data_gbg_summary_base = data_gbg_summary_list[-1] if i > 0 else init_data_gbg_summary_base
            data_gbg_stats_item = data_gbg_stats_list[i]
            data_gbg_summary_item = {'game_date': data_gbg_stats_item['game_date'],
                                     'home_team': data_gbg_stats_item['home_team'],
                                     'away_team': data_gbg_stats_item['away_team']}
            for item_number in range(4, len(item_names)):
                if '%' in item_names[item_number] or '/GP' in item_names[item_number]:
                    continue
                data_new = data_gbg_summary_base.get(item_names[item_number]) + \
                           data_gbg_stats_item.get(item_names[item_number])
                data_gbg_summary_item.update({item_names[item_number]: data_new})
            data_gbg_summary_list.append(data_gbg_summary_item)
        player_stats_summary.update({player_name: data_gbg_summary_list})

    playerIndex_stats_summary = {}
    for player_name in player_stats_summary.keys():
        # "{'id': match_id, 'position': player_position_basic, 'index': player_index_basic}"
        data_gbg_summary_list = player_stats_summary.get(player_name)
        player_info = match_name_info_dict.get(player_name)
        if player_info is None:
            print(player_name)
            continue
        playerIndex_stats_summary.update({player_info['index']: {'player_name': player_name, 'id': player_info['id'],
                                                                 'position': player_info['position'],
                                                                 'gbg_summary_list': data_gbg_summary_list}})

    with open("../resource/ice_hockey_201819/" + 'NHL_players_game_summary_201819.csv', 'w') as outfile:
        json.dump(playerIndex_stats_summary, outfile)

    return player_stats_summary


if __name__ == '__main__':
    player_basic_info_dir = '../resource/ice_hockey_201819/player_info_2018_2019.json'
    player_data_stats_dir = '/Local-Scratch/oschulte/Galen/Ice-hockey-data/player_stats/NHL_game_by_game_stats.csv'
    player_stats_gbg_all, item_names = collect_player_stats_by_date(player_data_stats_dir=player_data_stats_dir)
    player_all = player_stats_gbg_all.keys()
    match_name_info_dict = match_player_name_NHLcom(player_basic_info_dir, player_names=player_all)
    player_stats_summary = aggregate_player_stats_by_date(player_stats_gbg_all, item_names, match_name_info_dict)
    # with open("../resource/ice_hockey_201819/" + 'NHL_players_game_summary_201819.csv', 'r') as outfile:
    #     player_stats_summary = json.load(outfile)
    print('rescaling ...')
    horizontal_rescale(player_stats_summary, item_names, match_name_info_dict)

