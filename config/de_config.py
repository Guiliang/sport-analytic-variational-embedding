import yaml
from support.config_tools import InitWithDict

"""Deterministic Encoding Config"""


class DECongfig(object):
    def __init__(self, init):
        self.Learn = DECongfig.Learn(init["Learn"])
        self.Arch = DECongfig.Arch(init["Arch"])

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
        predict_target = None

    class Arch(InitWithDict):
        def __init__(self, init):
            self.LSTM = DECongfig.Arch.LSTM(init["LSTM"])
            self.Encode = DECongfig.Arch.Encode(init["Encode"])
            self.Dense = DECongfig.Arch.Dense(init["Dense"])

        class LSTM(InitWithDict):
            h_size = None
            lstm_layer_num = None

        class Encode(InitWithDict):
            latent_size = None
            label_size = None

        class Dense(InitWithDict):
            dense_layer_num = None
            output_layer_size = None
            apply_softmax = None
            hidden_node_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return DECongfig(config)
