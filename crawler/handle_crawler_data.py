import csv
import json


def aggregate_player_stats_by_date(player_data_stats_dir):
    item_names = None
    player_stats_summary = {}
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

                data_gbg_summary_list = player_stats_summary.get(player_name)
                if data_gbg_summary_list is None:
                    data_gbg_summary_base = {'game_date': 0, 'home_team': home_team, 'away_team': away_team}
                    for item_number in range(4, len(data_row)):
                        data_gbg_summary_base.update({item_names[item_number]: float(0)})
                    player_stats_summary[player_name] = [data_gbg_summary_base]
                else:
                    data_gbg_summary_base = data_gbg_summary_list[-1]
                data_gbg_summary_new = {'game_date': game_date, 'home_team': home_team, 'away_team': away_team}
                for item_number in range(4, len(data_row)):
                    data_new = data_gbg_summary_base.get(item_names[item_number]) + float(data_row[item_number])
                    data_gbg_summary_new.update({item_names[item_number]: data_new})
                player_stats_summary[player_name].append(data_gbg_summary_new)
                # break

            print(data_gbg_summary_new)

    with open("../resource/ice_hockey_201819/" + 'NHL_players_game_summary_201819.csv', 'w') as outfile:
        json.dump(player_stats_summary, outfile)


if __name__ == '__main__':
    with open("../resource/ice_hockey_201819/" + 'NHL_players_game_summary_201819.csv', 'r') as outfile:
        tmp = json.load(outfile)

    player_data_stats_dir = '/Local-Scratch/oschulte/Galen/Ice-hockey-data/player_stats/NHL_game_by_game_stats.csv'
    aggregate_player_stats_by_date(player_data_stats_dir=player_data_stats_dir)
    pass
