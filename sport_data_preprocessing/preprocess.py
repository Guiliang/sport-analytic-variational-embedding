import json
import os
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from data_config import action_all, interested_raw_features, interested_compute_features, teamList


class Preprocess:
    def __init__(self, hockey_data_dir, save_data_dir):
        self.hockey_data_dir = hockey_data_dir
        self.save_data_dir = save_data_dir

    def get_events(self, data_dir):
        with open(self.hockey_data_dir + data_dir) as f:
            data = json.load(f)
            events = data.get('events')
        return events

    def get_player_name(self, data_dir):
        players_info_dict = {}
        with open(self.hockey_data_dir + data_dir) as f:
            data = json.load(f)
            rosters = data.get('rosters')
        for teamId in rosters.keys():
            players = rosters.get(teamId)
            if len(players) == 0:
                continue
            for player_info in players:
                first_name = player_info.get('firstName')
                last_name = player_info.get('lastName')
                position = player_info.get('position')
                id = player_info.get('id')
                players_info_dict.update(
                    {int(id): {'first_name': first_name, 'last_name': last_name, 'position': position,
                               'teamId': int(teamId)}})

        return players_info_dict

    def generate_player_information(self, store_player_info_dir):
        files_all = os.listdir(self.hockey_data_dir)
        players_info_dict_all = {}
        for file in files_all:
            print("### handling player name in file: ", file)
            if file == '.Rhistory' or file == '.DS_Store':
                continue
            players_info_dict = self.get_player_name(file)
            players_info_dict_all.update(players_info_dict)

        player_global_index = 0
        for player_id in players_info_dict_all.keys():
            player_info = players_info_dict_all.get(player_id)
            player_info.update({'index': player_global_index})
            # players_info_dict_all.update({player_id: player_info})
            player_global_index += 1

        with open(store_player_info_dir, 'w') as f:
            json.dump(players_info_dict_all, f)

    def action_one_hot(self, action):
        one_hot = [0] * len(action_all)  # total 33 action
        idx = action_all.index(action)
        one_hot[idx] = 1
        return one_hot

    def team_one_hot(self, teamId):
        teamId_int = int(teamId)
        one_hot = [0] * 31  # total 31 team
        idx = teamList.index(teamId_int)
        one_hot[idx] = 1
        return one_hot

    def home_away_one_hot(self, home_away):
        one_hot = [0] * 2
        if home_away == 'H':
            one_hot[0] = 1
        elif home_away == 'A':
            one_hot[1] = 1
        return one_hot

    def get_reward(self, events, idx):
        gameTime_now = events[idx].get('gameTime')

    def get_duration(self, events, idx):
        duration = float(0)
        gameTime_now = events[idx].get('gameTime')
        if idx == len(events) - 1:
            duration = 3600. - gameTime_now
        else:
            gameTime_next = events[idx + 1].get('gameTime')
            duration = gameTime_next - gameTime_now
        return duration

    def get_time_remain(self, events, idx):
        gameTime = events[idx].get('gameTime')
        return 3600. - gameTime

    def is_switch_possession(self, events, idx):  # compare with former timestamp
        switch = False
        if idx == 0:
            switch = False
        else:
            team_pre = events[idx - 1].get('teamId')
            team_now = events[idx].get('teamId')
            if team_pre == team_now:
                switch = False
            else:
                switch = True
        return switch

    def is_home_away(self, events, idx):
        xCoord = events[idx].get('xCoord')
        xAdjCoord = events[idx].get('xAdjCoord')
        if xCoord == xAdjCoord:
            return 'H'
        else:
            return 'A'

    def is_switch_home_away(self, events, idx):  # compare with next timestamp
        switch = False
        if idx == len(events) - 1:
            switch = False
        else:
            h_a_now = self.is_home_away(events, idx)
            h_a_next = self.is_home_away(events, idx + 1)
            if h_a_now == h_a_next:
                switch = False
            else:
                switch = True
        return switch

    def get_velocity(self, coord_next, coord_now, duration):
        v = (float(coord_next) - float(coord_now)) / float(duration)
        return v

    def compute_v_x(self, events, idx, duration):
        v_x = float(0)
        if idx == len(events) - 1 or duration == 0:
            v_x = float(0)
        else:
            coord_next = events[idx + 1].get('xAdjCoord')
            coord_now = events[idx].get('xAdjCoord')
            if self.is_switch_home_away(events, idx):
                coord_next = -coord_next
            v_x = self.get_velocity(coord_next, coord_now, duration)
        return v_x

    def compute_v_y(self, events, idx, duration):
        v_y = float(0)
        if idx == len(events) - 1 or duration == 0:
            v_y = float(0)
        else:
            coord_next = events[idx + 1].get('yAdjCoord')
            coord_now = events[idx].get('yAdjCoord')
            if self.is_switch_home_away(events, idx):
                coord_next = -coord_next
            v_y = self.get_velocity(coord_next, coord_now, duration)
        return v_y

    def compute_angle2gate(self, events, idx):
        x_goal = 89
        y_goal = 0
        xAdj = events[idx].get('xAdjCoord')
        yAdj = events[idx].get('yAdjCoord')
        tant = (y_goal - yAdj) / (x_goal - xAdj)
        return tant

    def process_game_events(self, events):
        rewards_game = []
        state_feature_game = []
        action_game = []
        team_game = []
        lt_game = []

        lt = 0
        # reward = []
        for idx in range(0, len(events)):
            event = events[idx]
            teamId = event.get('teamId')
            teamId = int(teamId)
            action = event.get('name')
            if self.is_switch_possession(events, idx):
                lt = 1
            else:
                lt = lt + 1

            action_one_hot_vector = self.action_one_hot(action)
            team_one_hot_vector = self.team_one_hot(teamId)
            features_all = []
            # add raw features
            for feature_name in interested_raw_features:
                feature_value = event.get(feature_name)
                if feature_name == 'manpowerSituation':
                    if feature_value == 'powerPlay':
                        features_all.append(1.)
                    elif feature_value == 'evenStrength':
                        features_all.append(0.)
                    elif feature_value == 'shortHanded':
                        features_all.append(-1.)
                elif feature_name == 'outcome':
                    if feature_value == 'successful':
                        features_all.append(1.)
                    elif feature_value == 'undetermined':
                        features_all.append(0.)
                    elif feature_value == 'failed':
                        features_all.append(-1.)
                else:
                    features_all.append(float(feature_value))
            # add compute features
            for feature_name in interested_compute_features:
                if feature_name == 'velocity_x':
                    duration = self.get_duration(events, idx)
                    v_x = self.compute_v_x(events, idx, duration)
                    features_all.append(v_x)
                elif feature_name == 'velocity_y':
                    duration = self.get_duration(events, idx)
                    v_y = self.compute_v_y(events, idx, duration)
                    features_all.append(v_y)
                elif feature_name == 'time_remain':
                    time_remain = self.get_time_remain(events, idx)
                    features_all.append(time_remain)
                elif feature_name == 'duration':
                    duration = self.get_duration(events, idx)
                    features_all.append(duration)
                elif feature_name == 'home_away':
                    h_a = self.is_home_away(events, idx)
                    home_away_one_hot_vector = self.home_away_one_hot(h_a)
                    features_all += home_away_one_hot_vector
                elif feature_name == 'angle2gate':
                    angle2gate = self.compute_angle2gate(events, idx)
                    features_all.append(angle2gate)
            if action == 'goal' and h_a == 'H':
                print ('home goal')
                rewards_game.append(1)
            elif action == 'goal' and h_a == 'A':
                print ('away goal')
                rewards_game.append(-1)
            else:
                rewards_game.append(0)
            # rewards_game.append(reward)
            state_feature_game.append(np.asarray(features_all))
            action_game.append(np.asarray(action_one_hot_vector))
            team_game.append(np.asarray(team_one_hot_vector))
            lt_game.append(lt)
        return state_feature_game, action_game, team_game, lt_game, rewards_game

    def scale_allgame_features(self):
        files_all = os.listdir(self.hockey_data_dir)
        features_allgame = []
        for file in files_all:
            print("### Scale file: ", file)
            if file == '.Rhistory' or file == '.DS_Store':
                continue
            events = self.get_events(file)
            state_feature_game, _, _, _, _ = self.process_game_events(events)
            features_allgame += state_feature_game

        scaler = preprocessing.StandardScaler().fit(np.asarray(features_allgame))
        print("### Scaler ###")
        print(scaler.mean_)
        print(scaler.scale_)
        return scaler

    def process_all(self, scaler):
        files_all = os.listdir(self.hockey_data_dir)
        wrong_files = []
        for file in files_all:
            if file == '.Rhistory' or file == '.DS_Store':
                continue
            file_name = file.split('.')[0]
            game_name = file_name.split('-')[0]
            save_game_dir = self.save_data_dir + '/' + game_name
            events = self.get_events(file)
            state_feature_game, action_game, team_game, lt_game, rewards_game = self.process_game_events(events)
            try:
                state_feature_game_scale = scaler.transform(state_feature_game)
            except:
                print 'skip wrong file {0}'.format(file)
                wrong_files.append(file)
                continue
            if not os.path.exists(save_game_dir):
                os.mkdir(save_game_dir)
            print('Processing file {0}'.format(file))
            # save data to mat
            sio.savemat(save_game_dir + "/" + "reward_" + file_name + ".mat", {'reward': np.asarray(rewards_game)})
            sio.savemat(save_game_dir + "/" + "state_feature_" + file_name + ".mat",
                        {'state_feature': np.asarray(state_feature_game_scale)})
            sio.savemat(save_game_dir + "/" + "action_" + file_name + ".mat", {'action': np.asarray(action_game)})
            sio.savemat(save_game_dir + "/" + "lt_" + file_name + ".mat", {'lt': np.asarray(lt_game)})
            sio.savemat(save_game_dir + "/" + "team_" + file_name + ".mat", {'team': np.asarray(team_game)})

        return wrong_files


if __name__ == '__main__':
    hockey_data_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/data-sample/'
    # hockey_data_dir = '/cs/oschulte/2019-icehockey-data/2018-2019/'
    save_data_dir = '/cs/oschulte/Galen/Ice-hockey-data/2018-2019'
    prep = Preprocess(hockey_data_dir=hockey_data_dir, save_data_dir=save_data_dir)
    scaler = prep.scale_allgame_features()
    prep.process_all(scaler)
