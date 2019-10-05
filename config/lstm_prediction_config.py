import yaml
from support.config_tools import InitWithDict


class LSTMPredictConfig(object):
    def __init__(self, init):
        self.Learn = LSTMPredictConfig.Learn(init["Learn"])
        self.Arch = LSTMPredictConfig.Arch(init["Arch"])

    class Learn(InitWithDict):
        feature_number = None
        save_mother_dir = None
        batch_size = None
        keep_prob = None
        learning_rate = None
        number_of_total_game = None
        max_seq_length = None
        feature_type = None
        iterate_num = None
        model_type = None
        apply_softmax = None
        predict_target = None
        player_Id_style = None
        sport = None
        apply_pid = None

    class Arch(InitWithDict):
        def __init__(self, init):
            # super(TTLSTMCongfig.Arch, self).__init__(init)
            self.LSTM = LSTMPredictConfig.Arch.LSTM(init["LSTM"])
            self.Dense = LSTMPredictConfig.Arch.Dense(init["Dense"])

        class LSTM(InitWithDict):
            h_size = None
            lstm_layer_num = None

        class Dense(InitWithDict):
            dense_layer_number = None
            dense_layer_size = None
            output_layer_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return LSTMPredictConfig(config)
