import yaml
from support.config_tools import InitWithDict

"""Deterministic Encoding Config"""


class DEQsCongfig(object):
    def __init__(self, init):
        self.Learn = DEQsCongfig.Learn(init["Learn"])
        self.Arch = DEQsCongfig.Arch(init["Arch"])

    class Learn(InitWithDict):
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
        player_Id_style = None
        sport = None

    class Arch(InitWithDict):
        def __init__(self, init):
            self.LSTM = DEQsCongfig.Arch.LSTM(init["LSTM"])
            self.Embed = DEQsCongfig.Arch.Embed(init["Embed"])
            self.Dense = DEQsCongfig.Arch.Dense(init["Dense"])

        class LSTM(InitWithDict):
            h_size = None
            lstm_layer_num = None
            feature_number = None

        class Embed(InitWithDict):
            embed_size = None
            embed_layer_num = None
            latent_size = None

        class Dense(InitWithDict):
            dense_layer_num = None
            output_layer_size = None
            apply_softmax = None
            hidden_node_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return DEQsCongfig(config)
