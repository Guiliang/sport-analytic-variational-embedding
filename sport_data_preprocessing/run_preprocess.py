from preprocess import Preprocess
from build_seq import process_seq_all

if __name__ == '__main__':
    test_flag = True
    if test_flag:
        hockey_data_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/data-sample/'
        save_data_dir = '/Users/liu/Desktop/Ice-hokcey-data-sample/feature-sample'
    else:
        hockey_data_dir = '/cs/oschulte/2019-icehockey-data/2018-2019/'
        save_data_dir = '/cs/oschulte/Galen/Ice-hockey-data/2018-2019'
    prep = Preprocess(hockey_data_dir=hockey_data_dir, save_data_dir=save_data_dir)
    # scaler = prep.scale_allgame_features()
    # wrong_files = prep.process_all(scaler)
    process_seq_all(save_data_dir=save_data_dir)
    # print 'wrong files skipped are {0}'.format(str(wrong_files))

    prep.generate_player_information(store_player_info_dir='../resource/player_info_2018_2019.json')
