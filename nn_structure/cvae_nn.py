import tensorflow as tf
from config.cvae_config import CVAECongfig


class CVAE_NN(object):
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.Learn.learning_rate
        # self.keep_prob = config.Learn.keep_prob
        self.init_placeholder()

        # self.sarsa_hidden_layer_num = config.Arch.Sarsa.layer_num
        # self.sarsa_hidden_node = config.Arch.Sarsa.n_hidden
        self.sarsa_output_node = config.Arch.Sarsa.output_node
        self.lstm_encoder_cell_all = []

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.Learn.learning_rate)
        self.build()
        self.saver = tf.train.Saver()

    def init_placeholder(self):

        self.x_ph = tf.placeholder(dtype=tf.float32, name="x_input", shape=[None, self.config.Arch.CVAE.x_dim])
        if self.config.Learn.apply_lstm:
            self.y_ph = tf.placeholder(dtype=tf.float32, name="y_input",
                                       shape=[None, self.config.Learn.max_seq_length,
                                              self.config.Arch.CVAE.y_dim])
        else:
            self.y_ph = tf.placeholder(dtype=tf.float32, name="y_input", shape=[None, self.config.Arch.CVAE.y_dim])

        self.train_flag_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='training_flag')
        self.sarsa_target_ph = tf.placeholder(dtype=tf.float32,
                                              shape=[None, 3], name='sarsa_target')
        self.score_diff_target_ph = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, 3], name='score_diff_target')
        self.predict_target_ph = tf.placeholder(dtype=tf.float32,
                                                shape=[None, self.config.Arch.Predict.output_node],
                                                name='predict_target')
        self.trace_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="trace-length")

    def build(self):

        if self.config.Learn.apply_lstm:
            with tf.variable_scope("LSTM_encoder"):
                for i in range(self.config.Learn.lstm_layer_num):
                    self.lstm_encoder_cell_all.append(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.config.Learn.h_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.05, 0.05)))

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.)
        with tf.variable_scope("cvae"):
            with tf.variable_scope("gaussian_MLP_encoder"):
                self.en_w0 = tf.get_variable('w0', [self.config.Arch.CVAE.x_dim + self.config.Arch.CVAE.y_dim,
                                                    self.config.Arch.CVAE.n_hidden], initializer=w_init)
                self.en_b0 = tf.get_variable('b0', [self.config.Arch.CVAE.n_hidden], initializer=b_init)
                self.en_w1 = tf.get_variable('w1', [self.config.Arch.CVAE.n_hidden, self.config.Arch.CVAE.n_hidden],
                                             initializer=w_init)
                self.en_b1 = tf.get_variable('b1', [self.config.Arch.CVAE.n_hidden], initializer=b_init)
                self.en_wo = tf.get_variable('wo', [self.config.Arch.CVAE.n_hidden,
                                                    self.config.Arch.CVAE.latent_dim * 2], initializer=w_init)
                self.en_bo = tf.get_variable('bo', [self.config.Arch.CVAE.latent_dim * 2], initializer=b_init)

            # TODO: figure out how to set term "reuse"
            with tf.variable_scope("bernoulli_MLP_decoder"):
                self.de_w0 = tf.get_variable('w0', [self.config.Arch.CVAE.latent_dim + + self.config.Arch.CVAE.y_dim,
                                                    self.config.Arch.CVAE.n_hidden], initializer=w_init)
                self.de_b0 = tf.get_variable('b0', [self.config.Arch.CVAE.n_hidden],
                                             initializer=b_init)
                self.de_w1 = tf.get_variable('w1', [self.config.Arch.CVAE.n_hidden,
                                                    self.config.Arch.CVAE.n_hidden], initializer=w_init)
                self.de_b1 = tf.get_variable('b1', [self.config.Arch.CVAE.n_hidden], initializer=b_init)
                self.de_wo = tf.get_variable('wo', [self.config.Arch.CVAE.n_hidden,
                                                    self.config.Arch.CVAE.x_dim], initializer=w_init)
                self.de_bo = tf.get_variable('bo', [self.config.Arch.CVAE.x_dim], initializer=b_init)

        if self.config.Learn.apply_lstm:
            validation_input_size = self.config.Arch.CVAE.latent_dim + self.config.Learn.h_size
        else:
            validation_input_size = self.config.Arch.CVAE.latent_dim + self.config.Arch.CVAE.y_dim

        with tf.variable_scope("sarsa"):
            self.sarsa_weight = []
            self.sarsa_bias = []

            for i in range(0, self.config.Arch.Sarsa.layer_num):
                with tf.name_scope("Dense_Layer_{0}".format(str(i))):
                    if i == 0:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [validation_input_size,
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
                                            [validation_input_size,
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
        with tf.variable_scope("predict"):
            self.predict_weight = []
            self.predict_bias = []

            for i in range(0, self.config.Arch.Predict.layer_num):
                with tf.name_scope("Dense_Layer_{0}".format(str(i))):
                    if i == 0:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [validation_input_size,
                                             self.config.Arch.Predict.n_hidden],
                                            initializer=w_init)
                    # if i == self.sarsa_hidden_layer_num - 1:
                    else:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [self.config.Arch.Predict.n_hidden,
                                             self.config.Arch.Predict.n_hidden],
                                            initializer=w_init)
                    b = tf.get_variable("bias_{0}".format(str(i)), [self.config.Arch.Predict.n_hidden],
                                        initializer=b_init)
                    self.predict_weight.append(w)
                    self.predict_bias.append(b)

            with tf.name_scope("output_Layer"):
                self.predict_output_weight = tf.get_variable('weight_out', [self.config.Arch.Predict.n_hidden,
                                                                            self.config.Arch.Predict.output_node],
                                                             initializer=w_init)
                self.predict_output_bias = tf.get_variable("bias_out", [self.config.Arch.Predict.output_node],
                                                           initializer=b_init)

    # Gaussian MLP as conditional encoder
    def gaussian_MLP_conditional_encoder(self, encoder_input):
        with tf.variable_scope("gaussian_MLP_encoder"):
            # concatenate condition and image
            input = tf.concat(axis=1, values=[self.x_ph, encoder_input])

            # 1st hidden layer
            h0 = tf.matmul(input, self.en_w0) + self.en_b0
            h0 = tf.nn.elu(h0)
            # h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            h1 = tf.matmul(h0, self.en_w1) + self.en_b1
            h1 = tf.nn.tanh(h1)
            # h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer
            # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
            gaussian_params = tf.matmul(h1, self.en_wo) + self.en_bo

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.config.Arch.CVAE.latent_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.config.Arch.CVAE.latent_dim:])

        return mean, stddev

    # Bernoulli MLP as conditional decoder
    def categorical_MLP_conditional_decoder(self, z, encoder_input):
        with tf.variable_scope("bernoulli_MLP_decoder"):
            # concatenate condition and latent vectors
            input = tf.concat(axis=1, values=[z, encoder_input])

            # 1st hidden layer
            h0 = tf.matmul(input, self.de_w0) + self.de_b0
            h0 = tf.nn.tanh(h0)
            # h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            h1 = tf.matmul(h0, self.de_w1) + self.de_b1
            h1 = tf.nn.elu(h1)
            # h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer-mean
            y = tf.nn.softmax(tf.matmul(h1, self.de_wo) + self.de_bo)

        return y

    def sarsa_value_function(self, z, input_):
        with tf.variable_scope("sarsa"):
            with tf.name_scope('sarsa-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.Sarsa.layer_num):
                    dense_input = tf.concat([input_, z], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.sarsa_weight[i]) + self.sarsa_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('sarsa-output-layer'):
                output = tf.matmul(dense_output, self.sarsa_output_weight) + self.sarsa_output_bias

        return output

    def score_diff_value_function(self, z, input_):
        with tf.variable_scope("score_diff"):
            with tf.name_scope('diff-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.ScoreDiff.layer_num):
                    dense_input = tf.concat([input_, z], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.score_diff_weight[i]) + self.score_diff_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('diff-output-layer'):
                output = tf.matmul(dense_output, self.score_diff_output_weight) + self.score_diff_output_bias

        return output

    def predict_value_function(self, z, input_):
        with tf.variable_scope("predict_"):
            with tf.name_scope('predict-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.Predict.layer_num):
                    dense_input = tf.concat([input_, z], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.predict_weight[i]) + self.predict_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('predict-output-layer'):
                output = tf.matmul(dense_output, self.predict_output_weight) + self.predict_output_bias

        return tf.nn.softmax(output)

    # Gateway
    def autoencoder(self):

        if self.config.Learn.apply_lstm:
            rnn_output = None
            for i in range(self.config.Learn.lstm_layer_num):
                rnn_input = self.y_ph if i == 0 else rnn_output
                rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                    inputs=rnn_input, cell=self.lstm_encoder_cell_all[i],
                    sequence_length=self.trace_lengths_ph, dtype=tf.float32,
                    scope='sarsa_rnn_{0}'.format(str(i)))
            outputs = tf.stack(rnn_output)
            # Hack to build the indexing and retrieve the right output.
            self.batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            self.index = tf.range(0, self.batch_size) * self.config.Learn.max_seq_length \
                         + (self.trace_lengths_ph - 1)
            rnn_last = tf.gather(tf.reshape(outputs, [-1, self.config.Learn.h_size]), self.index)
            input_ = rnn_last
            encoder_input = tf.gather(tf.reshape(self.y_ph, [-1, self.config.Arch.CVAE.y_dim]), self.index)
        else:
            input_ = self.y_ph
            encoder_input = self.y_ph

        with tf.variable_scope("cvae"):
            # encoding
            mu, sigma = self.gaussian_MLP_conditional_encoder(encoder_input)

            # sampling by re-parameterization technique
            z_encoder = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
            z_prior = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

            z = tf.where(tf.cast(self.train_flag_ph, tf.bool), x=z_encoder, y=z_prior)

            # decoding
            x_decoder = self.categorical_MLP_conditional_decoder(z, encoder_input)
            x_decoder = tf.clip_by_value(x_decoder, 1e-8, 1 - 1e-8)

            # ELBO
            marginal_likelihood_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                onehot_labels=self.x_ph,
                logits=x_decoder,
                reduction=tf.losses.Reduction.NONE))
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma)
                                                - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
            KL_divergence_loss = tf.reduce_mean(KL_divergence)

            # ELBO = marginal_likelihood - KL_divergence
            # minimize loss instead of maximizing ELBO
            # loss = -ELBO

        return tf.nn.softmax(x_decoder), z_encoder, marginal_likelihood_loss, KL_divergence_loss, input_

    def __call__(self):

        # TODO: check if we can use multi-GPU implementation when necessary
        self.x_, self.z, self.marginal_likelihood_loss, self.KL_divergence_loss, input_ = self.autoencoder()
        cvae_loss = self.KL_divergence_loss + self.marginal_likelihood_loss
        tvars_cvae = tf.trainable_variables(scope='cvae')
        for t in tvars_cvae:
            print ('cvae_var: ' + str(t.name))
        cvae_grads = tf.gradients(tf.reduce_mean(cvae_loss), tvars_cvae)
        self.train_cvae_op = self.optimizer.apply_gradients(zip(cvae_grads, tvars_cvae))

        self.q_values_sarsa = self.sarsa_value_function(self.z, input_)
        self.td_loss = tf.reduce_mean(tf.square(self.q_values_sarsa - self.sarsa_target_ph), axis=-1)
        self.td_avg_diff = tf.reduce_mean(tf.abs(self.q_values_sarsa - self.sarsa_target_ph), axis=-1)
        if self.config.Learn.integral_update_flag:
            tvars_sarsa = tf.trainable_variables()
        else:
            tvars_sarsa = tf.trainable_variables(scope='sarsa')
        for t in tvars_sarsa:
            print ('sarsa_var: ' + str(t.name))
        td_grads = tf.gradients(tf.reduce_mean(self.td_loss), tvars_sarsa)
        self.train_td_op = self.optimizer.apply_gradients(zip(td_grads, tvars_sarsa))

        self.q_values_diff = self.score_diff_value_function(self.z, input_)
        self.td_score_diff_loss = tf.reduce_mean(tf.square(self.q_values_diff - self.score_diff_target_ph), axis=-1)
        self.td_score_diff_diff = tf.reduce_mean(tf.abs(self.q_values_diff - self.score_diff_target_ph), axis=-1)
        if self.config.Learn.integral_update_flag:
            tvars_score_diff = tf.trainable_variables()
        else:
            tvars_score_diff = tf.trainable_variables(scope='score_diff')
        for t in tvars_score_diff:
            print ('score_diff: ' + str(t.name))
        td_diff_grads = tf.gradients(tf.reduce_mean(self.td_score_diff_loss), tvars_score_diff)
        self.train_diff_op = self.optimizer.apply_gradients(zip(td_diff_grads, tvars_score_diff))

        self.predict_output = self.predict_value_function(self.z, input_)
        self.predict_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.predict_target_ph,
                                                                           logits=self.predict_output,
                                                                           reduction=tf.losses.Reduction.NONE))
        if self.config.Learn.integral_update_flag:
            tvars_predict = tf.trainable_variables()
        else:
            tvars_predict = tf.trainable_variables(scope='predict')
        for t in tvars_predict:
            print ('predict: ' + str(t.name))
        predict_grads = tf.gradients(tf.reduce_mean(self.predict_loss), tvars_predict)
        self.train_predict_op = self.optimizer.apply_gradients(zip(predict_grads, tvars_predict))

    # Conditional Decoder (Generator)
    def decoder(self, z):
        x_ = self.categorical_MLP_conditional_decoder(z)
        return x_


if __name__ == '__main__':
    """test the model builder"""
    cvae_config_path = "../environment_settings/icehockey_cvae_config.yaml"
    cvae_config = CVAECongfig.load(cvae_config_path)
    cvae_nn = CVAE_NN(config=cvae_config)
    cvae_nn()
