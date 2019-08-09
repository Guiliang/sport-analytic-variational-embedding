import tensorflow as tf
from config.de_embed_config import DEEmbedCongfig


class TD_Prediction_Embed:
    def __init__(self,
                 config,
                 model_name="tt_lstm",
                 rnn_type="bp_last_step"):
        """
        init the model
        """
        self.embed_ph = None
        self.rnn_type = rnn_type
        self.model_name = model_name
        self.config = config

        self.rnn_input_ph = None
        self.trace_lengths_ph = None
        self.home_away_indicator_ph = None
        self.y_ph = None

        self.lstm_cell_all = []
        self.dense_layer_weights = []
        self.dense_layer_bias = []
        self.embed_layer_weights = []
        self.embed_layer_bias = []
        self.read_out = None

    def build(self):
        """
        define a shallow dynamic LSTM
        """
        # with tf.device('/gpu:0'):
        with tf.name_scope(self.model_name):
            with tf.name_scope("LSTM-layer"):
                for i in range(self.config.Arch.LSTM.lstm_layer_num):
                    self.lstm_cell_all.append(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.LSTM.h_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.05, 0.05)))
            with tf.name_scope("Embed_layer"):
                for i in range(self.config.Arch.Embed.embed_layer_num):
                    w_input_size = self.config.Arch.Embed.latent_size if i > 0 else self.config.Arch.Embed.embed_size
                    w_output_size = self.config.Arch.Embed.latent_size
                    self.embed_home_w = tf.get_variable('w_embed_{0}'.format(str(i)), [w_input_size, w_output_size],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                    self.embed_home_b = tf.Variable(tf.zeros([w_output_size]), name="b_embed_{0}".format(str(i)))

            with tf.name_scope("Dense_Layer"):
                for i in range(self.config.Arch.Dense.dense_layer_num):
                    w_input_size = self.config.Arch.Dense.hidden_node_size if i > 0 else \
                        self.config.Arch.LSTM.h_size + self.config.Arch.Embed.latent_size
                    w_output_size = self.config.Arch.Dense.hidden_node_size \
                        if i < self.config.Arch.Dense.dense_layer_num - 1 else self.config.Arch.Dense.output_layer_size
                    self.dense_layer_weights.append(tf.get_variable('w_dense_{0}'.format(str(i)),
                                                                    [w_input_size, w_output_size],
                                                                    initializer=tf.contrib.layers.xavier_initializer()))
                    self.dense_layer_bias.append(
                        tf.Variable(tf.zeros([w_output_size]), name="b_dense_{0}".format(str(i))))

    def call(self):
        """
        build the network
        :return:
        """
        # with tf.device('/gpu:0'):
        with tf.name_scope(self.model_name):
            with tf.name_scope("LSTM_layer"):
                rnn_output = None
                for i in range(self.config.Arch.LSTM.lstm_layer_num):
                    rnn_input = self.rnn_input_ph if i == 0 else rnn_output
                    rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                        inputs=rnn_input, cell=self.lstm_cell_all[i],
                        sequence_length=self.trace_lengths_ph, dtype=tf.float32,
                        scope=self.rnn_type + '_home_rnn_{0}'.format(str(i)))
                outputs = tf.stack(rnn_output)
                # Hack to build the indexing and retrieve the right output.
                self.batch_size = tf.shape(outputs)[0]
                # Start indices for each sample
                self.index = tf.range(0, self.batch_size) * self.config.Learn.max_seq_length + (self.trace_lengths_ph - 1)
                # Indexing
                rnn_last = tf.gather(tf.reshape(outputs, [-1, self.config.Arch.LSTM.h_size]), self.index)

            with tf.name_scope("Embed_layer"):
                self.home_embed_layer = tf.matmul(rnn_last, self.embed_home_w) + self.embed_home_b
                embed_output = None
                for i in range(self.config.Arch.Embed.embed_layer_num):
                    dense_input = self.embed_ph if i == 0 else embed_output
                    # dense_input = embed_layer
                    embed_output = tf.matmul(dense_input, self.dense_layer_weights[i]) + self.dense_layer_bias[i]

            with tf.name_scope('Dense_Layer'):
                dense_output = None
                for i in range(self.config.Arch.Dense.dense_layer_num):
                    dense_input = tf.concat([rnn_last, embed_output], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.dense_layer_weights[i]) + self.dense_layer_bias[i]
                    if i < self.config.Arch.Dense.dense_layer_num - 1:
                        dense_output = tf.nn.relu(embed_output, name='activation_{0}'.format(str(i)))
            if self.config.Learn.apply_softmax:
                self.read_out = tf.nn.softmax(embed_output)
            else:
                self.read_out = embed_output
            with tf.name_scope("cost"):
                self.cost = tf.reduce_mean(tf.square(self.y_ph - self.read_out))
                self.diff = tf.reduce_mean(tf.abs(self.y_ph - self.read_out))
            tf.summary.histogram('cost', self.cost)

            with tf.name_scope("train"):
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.config.Learn.learning_rate).minimize(self.cost)

    def initialize_ph(self):
        """
        initialize the place holder
        :return:
        """
        rnn_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Learn.max_seq_length,
                                                               self.config.Arch.LSTM.feature_number], name="rnn-input-ph")
        trace_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="trace-length")
        home_away_indicator_ph = tf.cast(tf.placeholder(dtype=tf.int32, shape=[None], name="indicator-ph"), tf.bool)
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Arch.Dense.output_layer_size])
        embed_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Arch.Embed.embed_layer_num])

        self.rnn_input_ph = rnn_input_ph
        self.trace_lengths_ph = trace_lengths_ph
        self.home_away_indicator_ph = home_away_indicator_ph
        self.y_ph = y_ph
        self.embed_ph = embed_ph


class DeterministicEmbedding:
    def __init__(self, config, is_probability):
        self.config = config
        self.is_probability = is_probability
        self.y_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.embed_label_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.feature_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.trace_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None])
        self.rnn_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Learn.max_seq_length,
                                                                    self.config.Arch.LSTM.feature_number])
        config = DEEmbedCongfig
        self.dense_layer_bias = []
        self.dense_layer_weights = []
        self.lstm_cell_all = []
        self.embed_w = None
        self.embed_b = None
        self.feature_layer_weights = []
        self.feature_layer_bias = []

    def build(self):
        """
        define a shallow dynamic LSTM
        """
        # with tf.device('/gpu:0'):
        with tf.name_scope(self.config.Learn.model_type):
            with tf.name_scope("LSTM_layers"):
                for i in range(self.config.Arch.LSTM.lstm_layer_num):
                    self.lstm_cell_all.append(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.LSTM.h_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.05, 0.05)))
                    # lstm_cell_tmp = tf.nn.rnn_cell.LSTMCell(num_units=256)

            with tf.name_scope("Embed_layers"):
                self.embed_w = tf.get_variable('w_embed_home', [self.config.Arch.Encode.label_size,
                                                                self.config.Arch.Encode.latent_size],
                                               initializer=tf.contrib.layers.xavier_initializer())
                self.embed_b = tf.Variable(tf.zeros([self.config.Arch.Encode.latent_size]), name="b_embed_home")

            with tf.name_scope("Feature_layers"):
                for i in range(self.config.Arch.Feature.feature_layer_num):
                    w_input_size = self.config.Arch.Feature.feature_size \
                        if i == 0 else self.config.Arch.Feature.hidden_node_size
                    w_output_size = self.config.Arch.Feature.hidden_node_size
                    self.feature_layer_weights.append(tf.get_variable('w{0}_xaiver'.format(str(i)),
                                                                      [w_input_size, w_output_size],
                                                                      initializer=tf.contrib.layers.xavier_initializer()))
                    self.feature_layer_bias.append(tf.Variable(tf.zeros([w_output_size]), name="b_{0}".format(str(i))))

            with tf.name_scope("Dense_layers"):
                for i in range(self.config.Arch.Dense.dense_layer_num):
                    if i == 0:
                        w_input_size = self.config.Arch.Feature.hidden_node_size + self.config.Arch.Encode.latent_size \
                                       + self.config.Arch.LSTM.h_size
                    else:
                        w_input_size = self.config.Arch.Dense.hidden_node_size
                    w_output_size = self.config.Arch.Dense.hidden_node_size \
                        if i < self.config.Arch.Dense.dense_layer_num - 1 else self.config.Arch.Dense.output_layer_size
                    self.dense_layer_weights.append(tf.get_variable('fw{0}_xaiver'.format(str(i)),
                                                                    [w_input_size, w_output_size],
                                                                    initializer=tf.contrib.layers.xavier_initializer()))
                    self.dense_layer_bias.append(tf.Variable(tf.zeros([w_output_size]), name="fb_{0}".format(str(i))))

    def __call__(self):
        with tf.name_scope(self.config.Learn.model_type):
            with tf.name_scope("LSTM_layers"):
                rnn_output = None
                for i in range(self.config.Arch.LSTM.lstm_layer_num):
                    rnn_input = self.rnn_input_ph if i == 0 else rnn_output
                    rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                        inputs=rnn_input, cell=self.lstm_cell_all[i],
                        sequence_length=self.trace_lengths_ph, dtype=tf.float32,
                        scope=self.config.Learn.model_type + '_home_rnn_{0}'.format(str(i)))

                outputs = tf.stack(rnn_output)
                # Hack to build the indexing and retrieve the right output.
                self.batch_size = tf.shape(outputs)[0]
                # Start indices for each sample
                self.index = tf.range(0, self.batch_size) * self.config.Learn.max_seq_length + (
                        self.trace_lengths_ph - 1)
                # Indexing
                rnn_last = tf.gather(tf.reshape(outputs, [-1, self.config.Arch.LSTM.h_size]), self.index)

            with tf.name_scope("Embed_layers"):
                self.embed_layer = tf.matmul(self.embed_label_ph, self.embed_w) + self.embed_b

            # embed_layer = tf.concat([self.home_embed_layer, self.away_embed_layer], axis=1)
            # embed_layer = self.home_embed_layer

            with tf.name_scope("Feature_layers"):
                feature_output = None
                for i in range(self.config.Arch.Dense.dense_layer_num):
                    feature_input = self.feature_input_ph if i == 0 else feature_output
                    # dense_input = embed_layer
                    feature_output = tf.matmul(feature_input, self.feature_layer_weights[i]) + self.feature_layer_bias[
                        i]
                    if i < self.config.Arch.Feature.feature_layer_num - 1:
                        feature_output = tf.nn.relu(feature_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('Dense_layers'):
                dense_output = None
                for i in range(self.config.Arch.Dense.dense_layer_num):
                    dense_input = tf.concat([feature_output, self.embed_layer, rnn_last],
                                            axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.dense_layer_weights[i]) + self.dense_layer_bias[i]
                    if i < self.config.Arch.Dense.dense_layer_num - 1:
                        dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            if self.is_probability:
                self.read_out = tf.nn.softmax(dense_output)
                with tf.variable_scope('cross_entropy'):
                    self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y_ph,
                                                                logits=dense_output)
            else:
                self.read_out = dense_output
                with tf.variable_scope('Norm2'):
                    self.cost = tf.reduce_mean(tf.square(self.read_out - self.y_ph))
            tf.summary.histogram('cost', self.cost)

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.Learn.learning_rate).minimize(
                    self.cost)


if __name__ == '__main__':
    de_config_path = "../icehockey_de_config.yaml"
    cvrnn_config = DEEmbedCongfig.load(de_config_path)
    DE = DeterministicEmbedding(config=cvrnn_config)
