import tensorflow as tf
from config.cvae_config import CVAECongfig


class Encoder_NN(object):
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.Learn.learning_rate
        # self.keep_prob = config.Learn.keep_prob
        self.init_placeholder()

        # self.sarsa_hidden_layer_num = config.Arch.Sarsa.layer_num
        # self.sarsa_hidden_node = config.Arch.Sarsa.n_hidden
        self.sarsa_output_node = config.Arch.Sarsa.output_node

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.Learn.learning_rate)
        self.build()
        self.saver = tf.train.Saver()

    def init_placeholder(self):
        self.input_ph = tf.placeholder(dtype=tf.float32, name="x_input", shape=[None, self.config.Arch.Encoder.input_dim])
        self.output_ph = tf.placeholder(dtype=tf.float32, name="y_input", shape=[None, self.config.Arch.Encoder.output_dim])
        self.sarsa_target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='sarsa_target')
        self.score_diff_target_ph = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, 3], name='score_diff_target')

    def build(self):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.)
        with tf.variable_scope("encoder"):
            self.en_w0 = tf.get_variable('w0', [self.config.Arch.Encoder.input_dim,
                                                self.config.Arch.Encoder.n_hidden], initializer=w_init)
            self.en_b0 = tf.get_variable('b0', [self.config.Arch.Encoder.n_hidden], initializer=b_init)
            self.en_w1 = tf.get_variable('w1', [self.config.Arch.Encoder.n_hidden,
                                                self.config.Arch.Encoder.n_hidden], initializer=w_init)
            self.en_b1 = tf.get_variable('b1', [self.config.Arch.Encoder.n_hidden], initializer=b_init)
            self.en_we = tf.get_variable('we', [self.config.Arch.Encoder.n_hidden,
                                                self.config.Arch.Encoder.embed_dim], initializer=w_init)
            self.en_be = tf.get_variable('we', [self.config.Arch.Encoder.embed_dim], initializer=b_init)
            self.en_wo = tf.get_variable('wo', [self.config.Arch.Encoder.embed_dim,
                                                self.config.Arch.Encoder.output_dim], initializer=w_init)
            self.en_bo = tf.get_variable('wo', [self.config.Arch.Encoder.output_dim], initializer=b_init)


        with tf.variable_scope("sarsa"):
            self.sarsa_weight = []
            self.sarsa_bias = []

            for i in range(0, self.config.Arch.Sarsa.layer_num):
                with tf.name_scope("Dense_Layer_{0}".format(str(i))):
                    if i == 0:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [self.config.Arch.CVAE.latent_dim + self.config.Arch.CVAE.y_dim,
                                             self.config.Arch.Sarsa.n_hidden],
                                            initializer=w_init)
                    # if i == self.sarsa_hidden_layer_num - 1:
                    else:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [self.config.Arch.Sarsa.n_hidden, self.config.Arch.Sarsa.n_hidden],
                                            initializer=w_init)
                    b = tf.get_variable("bias_{0}".format(str(i)), [self.config.Arch.Sarsa.n_hidden],
                                        initializer=b_init)
                    self.sarsa_weight.append(w)
                    self.sarsa_bias.append(b)

            with tf.name_scope("output_Layer"):
                self.sarsa_output_weight = tf.get_variable('weight_out', [self.config.Arch.Sarsa.n_hidden,
                                                                          self.config.Arch.Sarsa.output_node],
                                                           initializer=w_init)
                self.sarsa_output_bias = tf.get_variable("bias_out", [self.config.Arch.Sarsa.output_node],
                                                         initializer=b_init)

        with tf.variable_scope("score_diff"):
            self.score_diff_weight = []
            self.score_diff_bias = []

            for i in range(0, self.config.Arch.Sarsa.layer_num):
                with tf.name_scope("Dense_Layer_{0}".format(str(i))):
                    if i == 0:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [self.config.Arch.CVAE.latent_dim + self.config.Arch.CVAE.y_dim,
                                             self.config.Arch.ScoreDiff.n_hidden],
                                            initializer=w_init)
                    # if i == self.sarsa_hidden_layer_num - 1:
                    else:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [self.config.Arch.ScoreDiff.n_hidden, self.config.Arch.ScoreDiff.n_hidden],
                                            initializer=w_init)
                    b = tf.get_variable("bias_{0}".format(str(i)), [self.config.Arch.ScoreDiff.n_hidden],
                                        initializer=b_init)
                    self.score_diff_weight.append(w)
                    self.score_diff_bias.append(b)

            with tf.name_scope("output_Layer"):
                self.score_diff_output_weight = tf.get_variable('weight_out', [self.config.Arch.ScoreDiff.n_hidden,
                                                                               self.config.Arch.ScoreDiff.output_node],
                                                                initializer=w_init)
                self.score_diff_output_bias = tf.get_variable("bias_out", [self.config.Arch.ScoreDiff.output_node],
                                                              initializer=b_init)


    def sarsa_value_function(self, embedding):
        with tf.variable_scope("sarsa"):
            with tf.name_scope('sarsa-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.Sarsa.layer_num):
                    dense_input = tf.concat([self.input_ph, embedding], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.sarsa_weight[i]) + self.sarsa_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('sarsa-output-layer'):
                output = tf.matmul(dense_output, self.sarsa_output_weight) + self.sarsa_output_bias

        return output

    def score_diff_value_function(self, embedding):
        with tf.variable_scope("score_diff"):
            with tf.name_scope('diff-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.Sarsa.layer_num):
                    dense_input = tf.concat([self.input_ph, embedding], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.score_diff_weight[i]) + self.score_diff_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('diff-output-layer'):
                output = tf.matmul(dense_output, self.sarsa_output_weight) + self.sarsa_output_bias

        return output

    # Gateway
    def encoder(self):
        with tf.variable_scope("encoder"):
            encoder_dense1 = tf.matmul(self.input_ph, self.en_w0) + self.en_b0
            encoder_dense2 = tf.matmul(encoder_dense1, self.en_w1) + self.en_b1
            embedding = tf.matmul(encoder_dense2, self.en_we) + self.en_be
            output = tf.matmul(embedding, self.en_wo) + self.en_bo
            player_prediction_output = tf.nn.softmax(output)

            likelihood_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                onehot_labels=self.output_ph,
                logits=player_prediction_output,
                reduction=tf.losses.Reduction.NONE))

        return player_prediction_output, embedding, likelihood_loss

    def __call__(self):
        # TODO: check if we can use multi-GPU implementation when necessary
        self.player_prediction, self.embedding, self.likelihood_loss = self.encoder()
        encoder_loss = self.likelihood_loss
        tvars_encoder = tf.trainable_variables(scope='encoder')
        for t in tvars_encoder:
            print ('encoder_var: ' + str(t.name))
        encoder_grads = tf.gradients(tf.reduce_mean(encoder_loss), tvars_encoder)
        self.train_encoder_op = self.optimizer.apply_gradients(zip(encoder_grads, tvars_encoder))

        self.q_values_sarsa = self.sarsa_value_function(self.embedding)
        self.td_loss = tf.reduce_mean(tf.square(self.q_values_sarsa - self.sarsa_target_ph), axis=-1)
        self.td_avg_diff = tf.reduce_mean(tf.abs(self.q_values_sarsa - self.sarsa_target_ph), axis=-1)
        tvars_sarsa = tf.trainable_variables(scope='sarsa')
        for t in tvars_sarsa:
            print ('sarsa_var: ' + str(t.name))
        td_grads = tf.gradients(tf.reduce_mean(self.td_loss), tvars_sarsa)
        self.train_td_op = self.optimizer.apply_gradients(zip(td_grads, tvars_sarsa))

        self.q_values_diff = self.score_diff_value_function(self.embedding)
        self.td_score_diff_loss = tf.reduce_mean(tf.square(self.q_values_diff - self.score_diff_target_ph), axis=-1)
        self.td_score_diff_diff = tf.reduce_mean(tf.abs(self.q_values_diff - self.score_diff_target_ph), axis=-1)
        tvars_score_diff = tf.trainable_variables(scope='score_diff')
        for t in tvars_score_diff:
            print ('score_diff: ' + str(t.name))
        td_diff_grads = tf.gradients(tf.reduce_mean(self.td_score_diff_loss), tvars_score_diff)
        self.train_diff_op = self.optimizer.apply_gradients(zip(td_diff_grads, tvars_score_diff))


if __name__ == '__main__':
    """test the model builder"""
    cvae_config_path = "../environment_settings/icehockey_cvae_config.yaml"
    cvae_config = CVAECongfig.load(cvae_config_path)
    cvae_nn = Encoder_NN(config=cvae_config)
    cvae_nn()
