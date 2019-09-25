import yaml
from support.config_tools import InitWithDict


class CVAECongfig(object):
    def __init__(self, init):
        self.Learn = CVAECongfig.Learn(init["Learn"])
        self.Arch = CVAECongfig.Arch(init["Arch"])

    class Learn(InitWithDict):
        keep_prob = None
        learning_rate = None
        number_of_total_game = None
        player_Id_style = None
        sport = None

    class Arch(InitWithDict):
        def __init__(self, init):
            # super(TTLSTMCongfig.Arch, self).__init__(init)
            self.CVAE = CVAECongfig.Arch.CVAE(init["CVAE"])
            self.Sarsa = CVAECongfig.Arch.Sarsa(init["Sarsa"])
            self.ScoreDiff = CVAECongfig.Arch.ScoreDiff(init["ScoreDiff"])

        class CVAE(InitWithDict):
            n_hidden = None
            latent_dim = None
            x_dim = None
            y_dim = None

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
        return CVAECongfig(config)
