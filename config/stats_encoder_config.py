import yaml
from support.config_tools import InitWithDict


class EncoderConfig(object):
    def __init__(self, init):
        self.Learn = EncoderConfig.Learn(init["Learn"])
        self.Arch = EncoderConfig.Arch(init["Arch"])

    class Learn(InitWithDict):
        keep_prob = None
        learning_rate = None
        number_of_total_game = None
        player_Id_style = None
        sport = None

    class Arch(InitWithDict):
        def __init__(self, init):
            # super(TTLSTMCongfig.Arch, self).__init__(init)
            self.Encoder = EncoderConfig.Arch.Encoder(init["Encoder"])
            self.Sarsa = EncoderConfig.Arch.Sarsa(init["Sarsa"])
            self.ScoreDiff = EncoderConfig.Arch.ScoreDiff(init["ScoreDiff"])

        class Encoder(InitWithDict):
            n_hidden = None
            embed_dim = None
            input_dim = None
            output_dim = None

        class Sarsa(InitWithDict):
            layer_num = None
            n_hidden = None
            output_node = None

        class ScoreDiff(InitWithDict):
            layer_num = None
            n_hidden = None
            output_node = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return EncoderConfig(config)
