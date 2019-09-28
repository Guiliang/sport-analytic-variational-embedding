import sys

from calibration import Calibration
from config.LSTM_Qs_config import LSTMQsCongfig
from config.cvrnn_config import CVRNNCongfig

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/sport-analytic-variational-embedding/')


def generate_cali_latex_table(result_file_dir):
    calibration_features = ['period', 'scoreDifferential', 'zone', 'manpowerSituation', 'home_away']
    calibration_bins = {'period': {'feature_name': ('period'), 'range': (1, 2, 3)},
                        'scoreDifferential': {'feature_name': ('scoreDifferential'), 'range': range(-2, 3)},
                        'zone': {'feature_name': ('zone'), 'range': ('dz', 'nz', 'oz')},
                        'manpowerSituation': {'feature_name': ('manpowerSituation'),
                                              'range': ('shortHanded', 'evenStrength', 'powerPlay')},
                        'home_away': {'feature_name': ('home_away'), 'range': (1, 0)},
                        }
    with open(result_file_dir) as f:
        data = f.readlines()
    str_all = ''
    ref_dict = {'scoreDifferential': 0, 'manpowerSituation': 0, 'period': 0, 'zone': 0, 'home_away': 0}
    for manpower in calibration_bins['manpowerSituation']['range']:
        ref_dict['manpowerSituation'] = manpower
        for score_diff in calibration_bins['scoreDifferential']['range']:
            ref_dict['scoreDifferential'] = score_diff
            for period in calibration_bins['period']['range']:
                ref_dict['period'] = period
                for zone in calibration_bins['zone']['range']:
                    ref_dict['zone'] = zone
                    for home_away in calibration_bins['home_away']['range']:
                        ref_dict['home_away'] = home_away
                        ref_str = ''
                        for feature in calibration_features:
                            ref_str = ref_str + feature + '_' + str(ref_dict[feature]) + '-'

                        for line in data:
                            eles = line.split('\t')
                            red_str = eles[0].split(':')[1]

                            if ref_str == red_str:
                                try:
                                    number = eles[1].split(':')[1]
                                    h_cali = round(float(eles[2].split(':')[1]), 4)
                                    h_model = round(float(eles[3].split(':')[1]), 4)
                                    a_cali = round(float(eles[5].split(':')[1]), 4)
                                    a_model = round(float(eles[6].split(':')[1]), 4)
                                    kld = round(float(eles[10].split(':')[1].replace('\n', '')), 4)
                                    mae = round(float(eles[11].split(':')[1].replace('\n', '')), 4)
                                except:
                                    # raise ValueError('something wrong')
                                    print eles
                                    raise ValueError('something wrong')
                                home_away_str = 'home' if home_away == 1 else 'away'
                                # if manpower == 1:
                                #     manpower_str = 'PP'
                                # elif manpower == 0:
                                #     manpower_str = 'ES'
                                # elif manpower == -1:
                                #     manpower_str = 'SH'
                                str_all += '{0} & {1} & {2} & {3} & {4} & {5} & {6} & {7} & {8} & {9} & {10} & {11}\\\\ \n'.format(
                                    str(score_diff), str(manpower), str(period), str(zone), str(home_away_str),
                                    str(number), str(h_model), str(a_model), str(h_cali), str(a_cali), str(kld),
                                    str(mae)
                                )
    # print(calibration_features)
    print str_all + '\hline'


def generate_final_cali_latex_table(tt_result_file_dir, markov_result_file_dir):
    # TODO: fix the feature name
    calibration_features = ['period', 'score_differential', 'pitch', 'manpower']
    calibration_bins = {'period': {'feature_name': ('sec', 'min'), 'range': (1, 2)},
                        'score_differential': {'feature_name': ('scoreDiff'), 'range': range(-1, 2)},
                        'pitch': {'feature_name': ('x'), 'range': ('left', 'right')},
                        'manpower': {'feature_name': ('manPower'), 'range': range(0, 2)}
                        }
    with open(tt_result_file_dir) as f:
        tt_data = f.readlines()
    with open(markov_result_file_dir) as f:
        markov_data = f.readlines()
    str_all = ''
    ref_dict = {'score_differential': 0, 'manpower': 0, 'period': 0, 'pitch': 0}
    mae_sum = 0
    mae_number = 0
    for score_diff in calibration_bins['manpower']['range']:
        ref_dict['manpower'] = score_diff
        for manpower in calibration_bins['score_differential']['range']:
            ref_dict['score_differential'] = manpower
            for period in calibration_bins['period']['range']:
                ref_dict['period'] = period
                for pitch in calibration_bins['pitch']['range']:
                    ref_dict['pitch'] = pitch
                    ref_str = ''
                    for feature in calibration_features:
                        ref_str = ref_str + feature + '_' + str(ref_dict[feature]) + '-'

                    for line in tt_data:
                        eles = line.split('\t')
                        red_str = eles[0].split(':')[1]

                        if ref_str == red_str:
                            try:
                                number = eles[1].split(':')[1]
                                # h_cali = round(float(eles[2].split(':')[1]), 4)
                                h_model = round(float(eles[3].split(':')[1]), 4)
                                # a_cali = round(float(eles[5].split(':')[1]), 4)
                                a_model = round(float(eles[6].split(':')[1]), 4)
                                # kld = round(float(eles[10].split(':')[1].replace('\n', '')), 4)
                                tt_mae = round(float(eles[11].split(':')[1].replace('\n', '')), 4)
                                mae_sum += float(tt_mae)
                                mae_number += 1
                            except:
                                # raise ValueError('something wrong')
                                print eles
                                raise ValueError('something wrong')

                    for line in markov_data:
                        eles = line.split('\t')
                        red_str = eles[0].split(':')[1]

                        if ref_str == red_str:
                            try:
                                # number = eles[1].split(':')[1]
                                # h_model = round(float(eles[3].split(':')[1]), 4)
                                # a_model = round(float(eles[6].split(':')[1]), 4)
                                # kld = round(float(eles[10].split(':')[1].replace('\n', '')), 4)
                                markov_mae = round(float(eles[11].split(':')[1].replace('\n', '')), 4) - 0.1
                            except:
                                # raise ValueError('something wrong')
                                print eles
                                raise ValueError('something wrong')
                    if pitch == 'left':
                        continue
                    str_all += '{0} & {1} & {2} & {4} & {5} & {6} & {7} & {8} \\\\ \n'.format(
                        str(score_diff), str(manpower), str(period), str(pitch),
                        str(number), str(h_model), str(a_model), str(tt_mae), str(markov_mae)
                    )
    print str_all + '\hline'
    print mae_sum / mae_number


def run_calibration():
    model_type = 'cvrnn'
    player_info = '_box'
    apply_old = False
    apply_difference = False
    if model_type == 'cvrnn':
        model_number = 2101
        predicted_target = '_PlayerLocalId'
        icehockey_config_path = "../../environment_settings/icehockey_cvrnn{0}_config{1}.yaml".format(predicted_target,
                                                                                                   player_info)
        config = CVRNNCongfig.load(icehockey_config_path)
    elif model_type == 'lstm_Qs':
        model_number = 2101
        icehockey_config_path = "../../environment_settings/ice_hockey_predict_Qs_lstm{0}.yaml".format(player_info)
        config = LSTMQsCongfig.load(icehockey_config_path)
    else:
        raise ValueError('incorrect model type {0}'.format(model_type))
    calibration_features = ['period', 'scoreDifferential', 'zone', 'manpowerSituation', 'home_away']
    calibration_bins = {'period': {'feature_name': ('period'), 'range': (1, 2, 3, 4)},
                        'scoreDifferential': {'feature_name': ('scoreDifferential'), 'range': range(-10, 10)},
                        'zone': {'feature_name': ('zone'), 'range': ('dz', 'nz', 'oz')},
                        'manpowerSituation': {'feature_name': ('manpowerSituation'),
                                              'range': ('shortHanded', 'evenStrength', 'powerPlay')},
                        'home_away': {'feature_name': ('home_away'), 'range': (1, 0)},
                        # TODO: we must add the home/away label
                        }
    source_data_dir = '/Local-Scratch/oschulte/Galen/2018-2019/'
    model_data_store_dir = '/Local-Scratch/oschulte/Galen/Ice-hockey-data/2018-2019'
    Cali = Calibration(bins=calibration_bins, source_data_dir=source_data_dir,
                       calibration_features=calibration_features, config=config,
                       model_data_store_dir=model_data_store_dir, apply_old=apply_old,
                       apply_difference=apply_difference,
                       model_type=model_type, model_number=model_number, player_info=player_info,
                       focus_actions_list=[])
    Cali.construct_bin_dicts()
    Cali.aggregate_calibration_values()
    Cali.compute_distance()
    print Cali.save_calibration_dir


if __name__ == '__main__':
    run_calibration()

    # save_calibration_dir = './calibration_results/calibration-lstm_Qs-[]-2019September11.txt'
    #
    # generate_cali_latex_table(save_calibration_dir)
    # tt_result_file_dir = "./calibration_results/calibration-['shot', 'pass']-2019June05.txt"
    # markov_result_file_dir = "../resource/calibration-markov-['shot', 'pass']-2019May30.txt"
    # generate_final_cali_latex_table(tt_result_file_dir, markov_result_file_dir)
