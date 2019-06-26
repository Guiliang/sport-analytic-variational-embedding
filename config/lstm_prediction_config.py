import yaml
from support.config_tools import InitWithDict


class LSTMCongfig(object):
    def __init__(self, init):
        self.Learn = LSTMCongfig.Learn(init["Learn"])
        self.Arch = LSTMCongfig.Arch(init["Arch"])

    class Learn(InitWithDict):
        dense_layer_num = None
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
        embed_size = None
        output_layer_size = None
        apply_softmax = None
        predict_target = None

    class Arch(InitWithDict):
        def __init__(self, init):
            # super(TTLSTMCongfig.Arch, self).__init__(init)
            self.LSTM = LSTMCongfig.Arch.LSTM(init["LSTM"])

        class LSTM(InitWithDict):
            h_size = None
            lstm_layer_num = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return LSTMCongfig(config)
