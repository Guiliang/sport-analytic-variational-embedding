import tensorflow as tf
from config.cvae_config import CVAECongfig


class CVAE_NN(object):
    def __init__(self, config):
        self.sarsa_output_bias = None
        self.sarsa_output_weight = None
        self.sarsa_bias = None
        self.sarsa_weight = None
        self.de_bo = None
        self.de_wo = None
        self.de_b1 = None
        self.de_w1 = None
        self.de_b0 = None
        self.de_w0 = None
        self.en_bo = None
        self.en_wo = None
        self.en_b1 = None
        self.en_w1 = None
        self.en_b0 = None
        self.en_w0 = None
        self.learning_rate = config.Learn.learning_rate
        self.cvae_n_hidden = config.Arch.CVAE.n_hidden
        self.cvae_n_output = config.Arch.CVAE.n_output
        self.keep_prob = config.Learn.keep_prob
        self.init_placeholder()

        self.sarsa_hidden_layer_num = config.Arch.Sarsa.layer_num
        self.sarsa_hidden_node = config.Arch.Sarsa.n_hidden
        self.sarsa_output_node = config.Arch.Sarsa.output_node

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.build()
        self.call()
        self.saver = tf.train.Saver()

    def init_placeholder(self):
        self.x_t0_ph = tf.placeholder(dtype=tf.int32, name="x0 input", shape=[None, None])
        self.y_t0_ph = tf.placeholder(dtype=tf.float32, name="y0 input", shape=[None, None])
        self.x_t1_ph = tf.placeholder(dtype=tf.int32, name="x1 input", shape=[None, None])
        self.y_t1_ph = tf.placeholder(dtype=tf.float32, name="y1 input", shape=[None, None])
        self.reward_ph = tf.placeholder(dtype=tf.int32, name="y1 input", shape=[None, self.sarsa_output_node])

    def build(self):
        with tf.variable_scope("gaussian_MLP_encoder"):
            dim_y = int(self.y_t0_ph.get_shape()[1])
            dim_x = int(self.x_t0_ph.get_shape()[1])
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)
            self.en_w0 = tf.get_variable('w0', [dim_x + dim_y, self.cvae_n_hidden + dim_y], initializer=w_init)
            self.en_b0 = tf.get_variable('b0', [self.cvae_n_hidden + dim_y], initializer=b_init)
            self.en_w1 = tf.get_variable('w1', [self.cvae_n_hidden + dim_y, self.cvae_n_hidden], initializer=w_init)
            self.en_b1 = tf.get_variable('b1', [self.cvae_n_hidden], initializer=b_init)
            self.en_wo = tf.get_variable('wo', [self.cvae_n_hidden, self.cvae_n_output * 2], initializer=w_init)
            self.en_bo = tf.get_variable('bo', [self.cvae_n_output * 2], initializer=b_init)

        # TODO: figure out how to set term "reuse"
        with tf.variable_scope("bernoulli_MLP_decoder"):
            dim_y = int(self.y_t0_ph.get_shape()[1])
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)
            self.de_w0 = tf.get_variable('w0', [self.cvae_n_hidden + dim_y, self.cvae_n_hidden], initializer=w_init)
            self.de_b0 = tf.get_variable('b0', [self.cvae_n_hidden], initializer=b_init)
            self.de_w1 = tf.get_variable('w1', [self.cvae_n_hidden, self.cvae_n_hidden], initializer=w_init)
            self.de_b1 = tf.get_variable('b1', [self.cvae_n_hidden], initializer=b_init)
            self.de_wo = tf.get_variable('wo', [self.cvae_n_hidden, self.cvae_n_output], initializer=w_init)
            self.de_bo = tf.get_variable('bo', [self.cvae_n_output], initializer=b_init)

        with tf.variable_scope("sarsa_network"):
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.)
            dim_x = int(self.x_t0_ph.get_shape()[1])
            self.sarsa_weight = []
            self.sarsa_bias = []

            for i in range(0, self.sarsa_hidden_layer_num):
                with tf.name_scope("Dense_Layer_{0}".format(str(i))):
                    if i == 0:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [dim_x, self.sarsa_hidden_node], initializer=w_init)
                    # if i == self.sarsa_hidden_layer_num - 1:
                    else:
                        w = tf.get_variable('weight_{0}'.format(str(i)),
                                            [self.sarsa_hidden_node, self.sarsa_hidden_node], initializer=w_init)
                    b = tf.get_variable("bias_{0}".format(str(i)), [self.sarsa_hidden_node], initializer=b_init)
                    self.sarsa_weight.append(w)
                    self.sarsa_bias.append(b)

            with tf.name_scope("output_Layer"):
                self.sarsa_output_weight = tf.get_variable('weight', [self.sarsa_hidden_node, self.sarsa_output_node],
                                                           initializer=w_init)
                self.sarsa_output_bias = tf.get_variable("bias", [self.sarsa_output_node], initializer=b_init)

    # Gaussian MLP as conditional encoder
    def gaussian_MLP_conditional_encoder(self):
        with tf.variable_scope("gaussian_MLP_encoder"):
            # concatenate condition and image
            input = tf.concat(axis=1, values=[self.x_t0_ph, self.y_t0_ph])

            # 1st hidden layer
            h0 = tf.matmul(input, self.en_w0) + self.en_b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            h1 = tf.matmul(h0, self.en_w1) + self.en_b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer
            # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
            gaussian_params = tf.matmul(h1, self.en_wo) + self.en_bo

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.cvae_n_output]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.cvae_n_output:])

        return mean, stddev

    # Bernoulli MLP as conditional decoder
    def bernoulli_MLP_conditional_decoder(self, z):
        with tf.variable_scope("bernoulli_MLP_decoder"):
            # concatenate condition and latent vectors
            input = tf.concat(concat_dim=1, values=[z, self.y_t0_ph])

            # 1st hidden layer
            h0 = tf.matmul(input, self.de_w0) + self.de_b0
            h0 = tf.nn.tanh(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            h1 = tf.matmul(h0, self.de_w1) + self.de_b1
            h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer-mean
            y = tf.sigmoid(tf.matmul(h1, self.de_wo) + self.de_bo)

        return y

    def sarsa_value_function(self, look_ahead=False):
        with tf.variable_scope("sarsa_network"):
            with tf.name_scope('sarsa-dense-layer'):
                dense_output = None
                for i in range(self.sarsa_hidden_layer_num):
                    if look_ahead:
                        dense_input = self.x_t1_ph if i == 0 else dense_output
                    else:
                        dense_input = self.x_t0_ph if i == 0 else dense_output
                    # dense_input = embed_layer
                    dense_output = tf.matmul(dense_input, self.sarsa_weight[i]) + self.sarsa_bias[i]
                    # if i < self.sarsa_hidden_layer_num - 1:
                    dense_output = tf.nn.relu(dense_output, name='activation_{0}'.format(str(i)))

            with tf.name_scope('sarsa-output-layer'):
                output = tf.matmul(dense_output, self.sarsa_output_weight) + self.sarsa_output_bias

        return output

    # Gateway
    def autoencoder(self):
        # encoding
        mu, sigma = self.gaussian_MLP_conditional_encoder()

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        x_ = self.bernoulli_MLP_conditional_decoder(z)
        x_ = tf.clip_by_value(x_, 1e-8, 1 - 1e-8)

        # ELBO
        marginal_likelihood = tf.reduce_sum(self.x_t0_ph * tf.log(x_) + (1 - self.x_t0_ph) * tf.log(1 - x_), 1)
        # TODO: figure out if the x here has to be x_ph
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        marginal_likelihood_loss = -tf.reduce_mean(marginal_likelihood)
        KL_divergence_loss = tf.reduce_mean(KL_divergence)

        # ELBO = marginal_likelihood - KL_divergence
        # minimize loss instead of maximizing ELBO
        # loss = -ELBO

        return x_, z, marginal_likelihood_loss, KL_divergence_loss

    def call(self):
        # TODO: check if we can use multi-GPU implementation when necessary
        x_, z, self.marginal_likelihood_loss, self.KL_divergence_loss = self.autoencoder()
        self.q_t0_values = self.sarsa_value_function(look_ahead=False)
        self.q_t1_values = self.sarsa_value_function(look_ahead=True)
        self.td_loss = tf.reduce_mean(tf.square(self.q_t1_values+self.reward_ph-self.q_t0_values), axis=-1)
        total_loss = self.td_loss+self.KL_divergence_loss+self.marginal_likelihood_loss
        self.train_op = self.optimizer.minimize(loss=total_loss)

    # Conditional Decoder (Generator)
    def decoder(self, z):
        x_ = self.bernoulli_MLP_conditional_decoder(z)

        return x_


if __name__ == '__main__':
    """test the model builder"""
    cvae_config_path = "./cvae-config.yaml"
    cvae_config = CVAECongfig.load(cvae_config_path)
    cvae_nn = CVAE_NN(config=cvae_config)
