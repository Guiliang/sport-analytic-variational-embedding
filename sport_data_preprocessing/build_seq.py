import scipy.io as sio
import numpy as np
import os


def new_state_feature(s_f, lt):
    rows = s_f.shape[0]
    feature_num = s_f.shape[1]
    state_feature_new = np.zeros([rows, 10, feature_num])
    for i in range(rows):
        trace = lt[0][i]
        if trace > 10:
            trace = 10
        idx = trace - 1
        cnt = 0
        while (idx >= 0):
            state_feature_new[i, idx, :] = s_f[i - cnt, :]
            idx = idx - 1
            cnt = cnt + 1
    return state_feature_new


def process_all(hockey_data_dir):
    folder_all = os.listdir(hockey_data_dir)
    for folder in folder_all:
        folder_path = hockey_data_dir + '/' + folder
        file_all = os.listdir(folder_path)
        state_feature_path = ""
        lt_path = ""
        for file in file_all:
            if "state" in file:
                state_feature_path = folder_path + '/' + file
            if "lt" in file:
                lt_path = folder_path + '/' + file
        state_feature = sio.loadmat(state_feature_path)
        state_feature = state_feature["state_feature"]
        lt = sio.loadmat(lt_path)
        lt = lt["lt"]
        print(folder)
        state_feature_new = new_state_feature(state_feature, lt)
        sio.savemat(folder_path + '/' + 'state_feature_seq_' + folder + '.mat',
                    {'state_feature_seq': state_feature_new})


if __name__ == '__main__':
    hockey_data_dir = '/cs/oschulte/Galen/Ice-hockey-data/2018-2019'
    process_all(hockey_data_dir)
