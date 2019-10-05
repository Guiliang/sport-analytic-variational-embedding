import tensorflow as tf

from config.lstm_prediction_config import LSTMPredictConfig


class Td_Prediction_NN:
    def __init__(self,
                 config):
        """
        init the model
        """
        # config = LSTMCongfig.load(None)
        self.config = config

        self.rnn_input_ph = None
        self.trace_lengths_ph = None
        self.y_ph = None

        self.lstm_cell_all = []
        self.dense_layer_weights = []
        self.dense_layer_bias = []
        self.embed_w = None
        self.embed_b = None
        self.read_out = None
        self.train_op = None

    def build(self):
        """
        define a shallow dynamic LSTM
        """
        # with tf.device('/gpu:0'):
        with tf.name_scope("LSTM_layer"):
            for i in range(self.config.Arch.LSTM.lstm_layer_num):
                self.lstm_cell_all.append(
                    tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.LSTM.h_size, state_is_tuple=True,
                                            initializer=tf.random_uniform_initializer(-0.05, 0.05)))

        with tf.name_scope("Dense_layer"):
            for i in range(self.config.Arch.Dense.dense_layer_number):
                w_input_size = self.config.Arch.Dense.dense_layer_size if i > 0 else self.config.Arch.LSTM.h_size
                w_output_size = self.config.Arch.Dense.dense_layer_size if i < self.config.Arch.\
                    Dense.dense_layer_number - 1 else self.config.Arch.Dense.output_layer_size
                self.dense_layer_weights.append(tf.get_variable('w{0}_xaiver'.format(str(i)),
                                                                [w_input_size, w_output_size],
                                                                initializer=tf.contrib.layers.xavier_initializer()))
                self.dense_layer_bias.append(tf.Variable(tf.zeros([w_output_size]), name="b_{0}".format(str(i))))

    def call(self):
        """
        build the network
        :return:
        """
        # with tf.device('/gpu:0'):
        with tf.name_scope("LSTM_layer"):
            rnn_output = None
            for i in range(self.config.Arch.LSTM.lstm_layer_num):
                rnn_input = self.rnn_input_ph if i == 0 else rnn_output
                rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                    inputs=rnn_input, cell=self.lstm_cell_all[i],
                    sequence_length=self.trace_lengths_ph, dtype=tf.float32,
                    scope='rnn_{0}'.format(str(i)))
            outputs = tf.stack(rnn_output)
            # Hack to build the indexing and retrieve the right output.
            self.batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            self.index = tf.range(0, self.batch_size) * self.config.Learn.max_seq_length + (self.trace_lengths_ph - 1)
            # Indexing
            rnn_last = tf.gather(tf.reshape(outputs, [-1, self.config.Arch.LSTM.h_size]), self.index)


        with tf.name_scope('Dense_layer'):
            dense_output = None
            for i in range(self.config.Arch.Dense.dense_layer_number):
                dense_input = rnn_last if i == 0 else dense_output
                # dense_input = embed_layer
                dense_output = tf.matmul(dense_input, self.dense_layer_weights[i]) + self.dense_layer_bias[i]
                if i < self.config.Arch.Dense.dense_layer_number - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

        self.read_out = tf.nn.softmax(dense_output)
        with tf.variable_scope('cross_entropy'):
            self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y_ph,
                                                        logits=dense_output)
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.Learn.learning_rate).minimize(self.cost)

    def initialize_ph(self):
        """
        initialize the place holder
        :return:
        """
        rnn_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Learn.max_seq_length,
                                                               self.config.Learn.feature_number], name="rnn-input-ph")
        trace_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="trace-length")
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Arch.Dense.output_layer_size])

        self.rnn_input_ph = rnn_input_ph
        self.trace_lengths_ph = trace_lengths_ph
        self.y_ph = y_ph
