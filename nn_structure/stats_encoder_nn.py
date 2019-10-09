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
        self.lstm_encoder_cell_all = []
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.Learn.learning_rate)
        self.build()
        self.saver = tf.train.Saver()

    def init_placeholder(self):
        if self.config.Learn.apply_lstm:
            self.input_ph = tf.placeholder(dtype=tf.float32, name="input",
                                           shape=[None, self.config.Learn.max_seq_length,
                                                  self.config.Arch.Encoder.input_dim])
        else:
            self.input_ph = tf.placeholder(dtype=tf.float32, name="input",
                                           shape=[None, self.config.Arch.Encoder.input_dim])
        self.output_ph = tf.placeholder(dtype=tf.float32, name="output",
                                        shape=[None, self.config.Arch.Encoder.output_dim])
        self.sarsa_target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='sarsa_target')
        self.score_diff_target_ph = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, 3], name='score_diff_target')
        self.predict_target_ph = tf.placeholder(dtype=tf.float32,
                                                shape=[None, self.config.Arch.Predict.output_node],
                                                name='predict_target')
        self.trace_lengths_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="trace-length")

    def build(self):
        if self.config.Learn.apply_lstm:
            with tf.variable_scope("LSTM_encoder"):
                for i in range(self.config.Arch.Encoder.lstm_layer_num):
                    self.lstm_encoder_cell_all.append(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.Encoder.h_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.05, 0.05)))

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.)
        with tf.variable_scope("encoder"):
            if self.config.Learn.apply_lstm:
                encoder_input_size = self.config.Arch.Encoder.h_size
            else:
                encoder_input_size = self.config.Arch.Encoder.input_dim

            self.en_w0 = tf.get_variable('w0', [encoder_input_size,
                                                self.config.Arch.Encoder.n_hidden], initializer=w_init)
            self.en_b0 = tf.get_variable('b0', [self.config.Arch.Encoder.n_hidden], initializer=b_init)
            self.en_w1 = tf.get_variable('w1', [self.config.Arch.Encoder.n_hidden,
                                                self.config.Arch.Encoder.n_hidden], initializer=w_init)
            self.en_b1 = tf.get_variable('b1', [self.config.Arch.Encoder.n_hidden], initializer=b_init)
            self.en_we = tf.get_variable('we', [self.config.Arch.Encoder.n_hidden,
                                                self.config.Arch.Encoder.embed_dim], initializer=w_init)
            self.en_be = tf.get_variable('be', [self.config.Arch.Encoder.embed_dim], initializer=b_init)
            self.en_wo = tf.get_variable('wo', [self.config.Arch.Encoder.embed_dim,
                                                self.config.Arch.Encoder.output_dim], initializer=w_init)
            self.en_bo = tf.get_variable('bo', [self.config.Arch.Encoder.output_dim], initializer=b_init)

        with tf.variable_scope("sarsa"):
            self.sarsa_weight = []
            self.sarsa_bias = []

            for i in range(0, self.config.Arch.Sarsa.layer_num):
                with tf.name_scope("Dense_Layer_{0}".format(str(i))):
                    if i == 0:
                        if not self.config.Learn.apply_lstm:
                            w = tf.get_variable('weight_{0}'.format(str(i)),
                                                [
                                                    self.config.Arch.Encoder.embed_dim + self.config.Arch.Encoder.input_dim,
                                                    self.config.Arch.Sarsa.n_hidden],
                                                initializer=w_init)
                        else:
                            w = tf.get_variable('weight_{0}'.format(str(i)),
                                                [self.config.Arch.Encoder.embed_dim + self.config.Arch.Encoder.h_size,
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
                        if not self.config.Learn.apply_lstm:
                            w = tf.get_variable('weight_{0}'.format(str(i)),
                                                [
                                                    self.config.Arch.Encoder.embed_dim + self.config.Arch.Encoder.input_dim,
                                                    self.config.Arch.Sarsa.n_hidden],
                                                initializer=w_init)
                        else:
                            w = tf.get_variable('weight_{0}'.format(str(i)),
                                                [self.config.Arch.Encoder.embed_dim + self.config.Arch.Encoder.h_size,
                                                 self.config.Arch.Sarsa.n_hidden],
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

        with tf.variable_scope("prediction"):
            self.prediction_weight = []
            self.prediction_bias = []

            for i in range(0, self.config.Arch.Sarsa.layer_num):
                with tf.name_scope("Dense_Layer_{0}".format(str(i))):
                    if i == 0:
                        if not self.config.Learn.apply_lstm:
                            w = tf.get_variable('weight_{0}'.format(str(i)),
                                                [
                                                    self.config.Arch.Encoder.embed_dim + self.config.Arch.Encoder.input_dim,
                                                    self.config.Arch.Predict.n_hidden],
                                                initializer=w_init)
                        else:
                            w = tf.get_variable('weight_{0}'.format(str(i)),
                                                [
                                                    self.config.Arch.Encoder.embed_dim + self.config.Arch.Encoder.h_size,
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
                    self.prediction_weight.append(w)
                    self.prediction_bias.append(b)

            with tf.name_scope("output_Layer"):
                self.prediction_output_weight = tf.get_variable('weight_out', [self.config.Arch.Predict.n_hidden,
                                                                               self.config.Arch.Predict.output_node],
                                                                initializer=w_init)
                self.prediction_output_bias = tf.get_variable("bias_out", [self.config.Arch.Predict.output_node],
                                                              initializer=b_init)

    def sarsa_value_function(self, input_, embedding):
        with tf.variable_scope("sarsa"):
            with tf.name_scope('sarsa-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.Sarsa.layer_num):
                    dense_input = tf.concat([input_, embedding], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.sarsa_weight[i]) + self.sarsa_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('sarsa-output-layer'):
                output = tf.matmul(dense_output, self.sarsa_output_weight) + self.sarsa_output_bias

        return output

    def score_diff_value_function(self, input_, embedding):
        with tf.variable_scope("score_diff"):
            with tf.name_scope('diff-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.Sarsa.layer_num):
                    dense_input = tf.concat([input_, embedding], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.score_diff_weight[i]) + self.score_diff_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('diff-output-layer'):
                output = tf.matmul(dense_output, self.sarsa_output_weight) + self.sarsa_output_bias

        return output

    def prediction_value_function(self, input_, embedding):
        with tf.variable_scope("prediction"):
            with tf.name_scope('pred-dense-layer'):
                dense_output = None
                for i in range(self.config.Arch.Sarsa.layer_num):
                    dense_input = tf.concat([input_, embedding], axis=1) if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.prediction_weight[i]) + self.prediction_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('pred-output-layer'):
                output = tf.matmul(dense_output, self.prediction_output_weight) + self.prediction_output_bias

        return output

    # Gateway
    def encoder(self):
        with tf.variable_scope("LSTM_encoder"):
            if self.config.Learn.apply_lstm:
                rnn_output = None
                for i in range(self.config.Arch.Encoder.lstm_layer_num):
                    rnn_input = self.input_ph if i == 0 else rnn_output
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
                rnn_last = tf.gather(tf.reshape(outputs, [-1, self.config.Arch.Encoder.h_size]), self.index)
                input_ = rnn_last
            else:
                input_ = self.input_ph

        with tf.variable_scope("LSTM_encoder"):
            encoder_dense1 = tf.matmul(input_, self.en_w0) + self.en_b0
            encoder_dense2 = tf.matmul(encoder_dense1, self.en_w1) + self.en_b1
            embedding = tf.matmul(encoder_dense2, self.en_we) + self.en_be
            output = tf.matmul(embedding, self.en_wo) + self.en_bo
            player_prediction_output = tf.nn.softmax(output)

            likelihood_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                onehot_labels=self.output_ph,
                logits=player_prediction_output,
                reduction=tf.losses.Reduction.NONE))

        return player_prediction_output, embedding, likelihood_loss, input_

    def __call__(self):
        # TODO: check if we can use multi-GPU implementation when necessary
        self.player_prediction, self.embedding, self.likelihood_loss, input_ = self.encoder()
        encoder_loss = self.likelihood_loss
        tvars_encoder = tf.trainable_variables(scope='encoder')
        for t in tvars_encoder:
            print ('encoder_var: ' + str(t.name))
        encoder_grads = tf.gradients(tf.reduce_mean(encoder_loss), tvars_encoder)
        self.train_encoder_op = self.optimizer.apply_gradients(zip(encoder_grads, tvars_encoder))

        self.q_values_sarsa = self.sarsa_value_function(self.embedding, input_)
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

        self.q_values_diff = self.score_diff_value_function(self.embedding, input_)
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

        self.prediction_prob = self.prediction_value_function(self.embedding, input_)
        self.predict_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.predict_target_ph,
                                                                           logits=self.prediction_prob,
                                                                           reduction=tf.losses.Reduction.NONE))
        if self.config.Learn.integral_update_flag:
            tvars_prediction = tf.trainable_variables()
        else:
            tvars_prediction = tf.trainable_variables(scope='prediction')

        for t in tvars_prediction:
            print ('prediction: ' + str(t.name))
        prediction_grads = tf.gradients(tf.reduce_mean(self.predict_loss), tvars_prediction)
        self.train_prediction_op = self.optimizer.apply_gradients(zip(prediction_grads, tvars_prediction))


if __name__ == '__main__':
    """test the model builder"""
    cvae_config_path = "../environment_settings/icehockey_cvae_config.yaml"
    cvae_config = CVAECongfig.load(cvae_config_path)
    cvae_nn = Encoder_NN(config=cvae_config)
    cvae_nn()
