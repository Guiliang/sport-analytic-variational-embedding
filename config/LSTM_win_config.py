import yaml
from support.config_tools import InitWithDict


class LSTMWinCongfig(object):
    def __init__(self, init):
        self.Learn = LSTMWinCongfig.Learn(init["Learn"])
        self.Arch = LSTMWinCongfig.Arch(init["Arch"])

    class Learn(InitWithDict):
        save_mother_dir = None
        batch_size = None
        keep_prob = None
        learning_rate = None
        gaussian_size = None
        number_of_total_game = None
        max_seq_length = None
        feature_type = None
        iterate_num = None
        model_type = None
        predict_target = None
        player_Id_style = None
        sport = None
        apply_softmax = None
        data_name = None
        apply_rl = None

    class Arch(InitWithDict):
        def __init__(self, init):
            self.LSTM = LSTMWinCongfig.Arch.LSTM(init["LSTM"])
            self.Dense = LSTMWinCongfig.Arch.Dense(init["Dense"])

        class LSTM(InitWithDict):
            h_size = None
            lstm_layer_num = None
            feature_number = None

        class Dense(InitWithDict):
            dense_layer_num = None
            output_layer_size = None
            hidden_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return LSTMWinCongfig(config)
