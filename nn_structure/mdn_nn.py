import tensorflow as tf

from support.model_tools import normal_td, calc_pdf


class MixtureDensityNN():
    def __init__(self, config):
        self.lstm_cell_all = []
        self.config = config
        self.dense_layer_weights = []
        self.dense_layer_bias = []
        self._build()
        self._init_placeholder()
        assert self.config.Learn.gaussian_size == 1

    def _init_placeholder(self):
        """initialize place holder"""
        self.y_mu_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3],
                                      name="y_mu-ph")
        self.y_var_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3],
                                       name="y_var-ph")
        # self.y_pi_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Learn.gaussian_size, 3],
        #                               name="y_pi-ph")
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="r-ph")
        self.trace_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='trace_length_ph')
        self.rnn_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.config.Learn.max_seq_length,
                                                                    self.config.Arch.LSTM.feature_number],
                                           name="rnn_input_ph-ph")

    def _build(self):
        """initialize weights"""
        with tf.name_scope("LSTM_Layer"):
            for i in range(self.config.Arch.LSTM.lstm_layer_num):
                self.lstm_cell_all.append(
                    tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.LSTM.h_size, state_is_tuple=True,
                                            initializer=tf.random_uniform_initializer(-0.05, 0.05)))

        with tf.name_scope('Dense_Layer'):
            for i in range(self.config.Arch.Dense.dense_layer_num):
                w_input_size = self.config.Arch.LSTM.h_size if i == 0 else self.config.Arch.Dense.hidden_size
                # w_output_size = self.config.Arch.Dense.output_size if i == self.config.Arch.Dense.dense_layer_num - 1 \
                #     else self.config.Arch.Dense.hidden_size
                w_output_size = self.config.Arch.Dense.hidden_size
                self.dense_layer_weights.append(
                    tf.get_variable('w_embed_{0}'.format(str(i)), [w_input_size, w_output_size],
                                    initializer=tf.contrib.layers.xavier_initializer()))
                self.dense_layer_bias.append(tf.Variable(tf.zeros([w_output_size]), name="b_dense_{0}".format(str(i))))

        with tf.name_scope('Mu'):
            self.mu_weight = tf.get_variable('w_mu',
                                             {self.config.Arch.Dense.hidden_size,
                                              self.config.Learn.gaussian_size * 3},
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.mu_bias = tf.get_variable('b_mu', [self.config.Learn.gaussian_size * 3],
                                           initializer=tf.contrib.layers.xavier_initializer())
        with tf.name_scope('Var'):
            self.var_weight = tf.get_variable('w_var',
                                              {self.config.Arch.Dense.hidden_size,
                                               self.config.Learn.gaussian_size * 3},
                                              initializer=tf.contrib.layers.xavier_initializer())
            self.var_bias = tf.get_variable('b_var', [self.config.Learn.gaussian_size * 3],
                                            initializer=tf.contrib.layers.xavier_initializer())

            # with tf.name_scope('Pi'):
            #     self.pi_weight = tf.get_variable('w_pi',
            #                                      {self.config.Arch.Dense.hidden_size,
            #                                       self.config.Learn.gaussian_size * 3},
            #                                      initializer=tf.contrib.layers.xavier_initializer())
            #     self.pi_bias = tf.get_variable('b_pi', [self.config.Learn.gaussian_size * 3],
            #                                    initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self):
        """connect the network"""

        with tf.name_scope("LSTM_Layer"):
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

        with tf.name_scope('Dense_Layer'):
            dense_output = None
            for i in range(self.config.Arch.Dense.dense_layer_num):
                dense_input = rnn_last if i == 0 else dense_output
                dense_output = tf.matmul(dense_input, self.dense_layer_weights[i]) + self.dense_layer_bias[i]
                # if i < self.config.Arch.Dense.dense_layer_num - 1:
                dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

        with tf.name_scope('Mu'):
            self.mu = tf.matmul(dense_output, self.mu_weight) + self.mu_bias
            if self.config.Learn.apply_softmax:
                self.mu = tf.nn.softmax(self.mu)
            self.mu_out = self.mu
            # tf.reshape(self.mu,shape=[-1, self.config.Learn.gaussian_size, 3])
            # if self.config.Learn.apply_softmax:
            # self.mu = tf.reshape(tf.nn.softmax(self.mu_out), shape=[-1, self.config.Learn.gaussian_size*3])
            # self.mu_out = tf.reshape(self.mu,
            #                          shape=[-1, self.config.Learn.gaussian_size, 3])

        with tf.name_scope('Var'):
            var_output = tf.matmul(dense_output, self.var_weight) + self.var_bias
            self.var = tf.exp(var_output)  # it could be softplus as well
            self.var_out = self.var
            # self.var_out = tf.reshape(self.var, shape=[-1, self.config.Learn.gaussian_size, 3])

        # with tf.name_scope('Pi'):
        #     pi_output = tf.matmul(dense_output, self.pi_weight) + self.pi_bias
        #     self.pi = tf.nn.softmax(pi_output)  # TODO: the position of softmax is wrong
        #     self.pi_out = tf.reshape(self.pi, shape=[-1, self.config.Learn.gaussian_size, 3])

        with tf.name_scope('loss'):
            self.normal_loss = self._compute_normal_loss()
            self.td_loss, self.bias_loss = self._compute_de_loss()

        with tf.name_scope("train"):
            self.train_normal_step = tf.train.AdamOptimizer(
                learning_rate=self.config.Learn.learning_rate / 1000).minimize(self.normal_loss)
            self.train_pretrain_step = tf.train.AdamOptimizer(
                learning_rate=self.config.Learn.learning_rate).minimize(self.td_loss+self.bias_loss)

    def _compute_normal_loss(self):
        """MDN Loss Function
        """
        self.y_mu = self.y_mu_ph
        self.y_var = self.y_var_ph
        # self.y_pi = self.y_pi_ph
        # self.y_mu = tf.reshape(self.y_mu_ph, shape=[-1, self.config.Learn.gaussian_size * 3])
        # self.y_var = tf.reshape(self.y_var_ph, shape=[-1, self.config.Learn.gaussian_size * 3])
        # self.y_pi = tf.reshape(self.y_pi_ph, shape=[-1, self.config.Learn.gaussian_size * 3])
        self.r = tf.tile(self.r_ph, [1, self.config.Learn.gaussian_size])
        normal_diff = calc_pdf(y=self.y_mu, mu=self.mu, var=self.var)
        # normal_diff = normal_td(mu1=self.mu, mu2=self.y_mu, var1=self.var, var2=self.y_var, y=self.r)
        # multiply with each pi and sum it
        # out = tf.multiply(out, self.pi)
        # normal_diff = tf.reduce_sum(normal_diff, 1, keep_dims=True)
        out = -tf.log(normal_diff + 1e-10)  # negative log likelihood
        return tf.reduce_mean(out)

    def _compute_de_loss(self):
        """DE TD loss for pre-training"""
        self.y_mu = self.y_mu_ph
        td_loss = tf.square(self.y_mu + self.r_ph - self.mu)

        bias_punish_loss = (self.mu[:, 0] / self.mu[:, 1]) + (self.mu[:, 1] / self.mu[:, 0])

        return tf.reduce_mean(td_loss), tf.reduce_mean(bias_punish_loss/1e5)
