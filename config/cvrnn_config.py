import yaml
from support.config_tools import InitWithDict


class CVRNNCongfig(object):
    def __init__(self, init):
        self.Learn = CVRNNCongfig.Learn(init["Learn"])
        self.Arch = CVRNNCongfig.Arch(init["Arch"])

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
        action_number = None
        predict_target = None
        gamma = None
        player_cluster_number = None
        data_name = None
        player_Id_style = None
        sport = None

    class Arch(InitWithDict):
        def __init__(self, init):
            # super(TTLSTMCongfig.Arch, self).__init__(init)
            self.CVRNN = CVRNNCongfig.Arch.CVRNN(init["CVRNN"])
            self.SARSA = CVRNNCongfig.Arch.SARSA(init["SARSA"])

        class CVRNN(InitWithDict):
            hidden_dim = None
            latent_dim = None
            x_dim = None
            y_dim = None
            z_dim = None

        class SARSA(InitWithDict):
            dense_layer_number = None
            dense_layer_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return CVRNNCongfig(config)
