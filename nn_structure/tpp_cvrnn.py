import collections

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from config.tpp_cvrnn_config import TPPCVRNNConfig


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


_TPPCVRNNStateTuple = collections.namedtuple("tpp_cvrnn_state_tuple", ("c", "h", 'mu_p', 'sigma_p'))


class TPPCVRNNStateTuple(_TPPCVRNNStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, mu_p, sigma_p) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


class TemporalProcessCVRNNCell(tf.nn.rnn_cell.RNNCell):
    """Variational RNN cell."""

    # @property
    def __init__(self, x_dim, y_dim, h_dim, z_dim=100, output_dim_list=[], **kwargs):
        super(TemporalProcessCVRNNCell, self).__init__(**kwargs)
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_y = y_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_y_1 = y_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim
        # self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        self.lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.n_h, state_is_tuple=True)
        self.output_dim_list = [self.n_z, self.n_z, self.n_x, self.n_x, self.n_x, self.n_z, self.n_z, self.n_z]

    @property
    def state_size(self):
        return (self.n_h, self.n_h, self.n_z, self.n_z)

    @property
    def output_size(self):
        # enc_mu, enc_sigma, dec_mu, dec_sigma, dec_x, prior_mu, prior_sigma
        return sum(self.output_dim_list)
        # return self.n_h

    def __call__(self, input, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, m, last_prior_mu, last_prior_sigma, = state  # TODO: why shall we apply c instead of m
            x, y, last_input_prior_mu, last_input_prior_sigma, train_flag_ph = \
                tf.split(value=input,
                         num_or_size_splits=[self.n_x, self.n_y, self.n_z, self.n_z, 1], axis=1)
            train_flag_ph = tf.cast(tf.squeeze(train_flag_ph), tf.bool)

            with tf.variable_scope("phi_y"):
                y_phi = linear(y, self.n_h)

            with tf.variable_scope("Prior"):
                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(tf.concat(values=[y_phi, m], axis=1), self.n_prior_hidden))
                with tf.variable_scope("delta_mu"):
                    delta_prior_mu = linear(prior_hidden, self.n_z)
                with tf.variable_scope("mu"):
                    prior_mu = delta_prior_mu + last_prior_mu + last_input_prior_mu
                    # last_input_prior_mu is a zero matrix except the first time step
                with tf.variable_scope("delta_sigma"):
                    delta_prior_sigma = linear(prior_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    prior_sigma = delta_prior_sigma + last_prior_sigma + last_input_prior_sigma
                    # last_input_prior_sigma is a zero matrix except the first time step
                    prior_sigma = tf.nn.softplus(prior_sigma)

            # lambda_prior = tf.nn.relu(linear(tf.concat(values=[y_phi, m], axis=1), self.n_prior_hidden))

            with tf.variable_scope("cond_x"):
                xy = tf.concat(values=(x, linear(y, self.n_h)), axis=1)
            with tf.variable_scope("phi_x"):
                xy_phi = tf.nn.relu(linear(xy, self.n_h))

            with tf.variable_scope("Encoder"):
                with tf.variable_scope("hidden"):
                    enc_hidden = tf.nn.relu(linear(tf.concat(axis=1, values=(xy_phi, m)), self.n_enc_hidden))
                with tf.variable_scope("mu"):
                    enc_mu = linear(enc_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.nn.softplus(linear(enc_hidden, self.n_z))
            # print x.get_shape().as_list()
            # eps = tf.random_normal((x.get_shape().as_list()[0], self.n_z), 0.0, 1.0, dtype=tf.float32)
            eps1 = tf.random_normal((tf.shape(x)[0], self.n_z), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon
            z_encoder = tf.add(enc_mu, tf.multiply(enc_sigma, eps1))
            z_prior = tf.add(prior_mu, tf.multiply(prior_sigma, eps1))
            with tf.variable_scope("cond_z"):
                z = tf.where(train_flag_ph, x=z_encoder, y=z_prior)
                zy = tf.concat(values=(z, linear(y, self.n_h)), axis=1)
            with tf.variable_scope("Phi_z"):
                zy_phi = tf.nn.relu(linear(zy, self.n_h))

            with tf.variable_scope("Decoder"):
                with tf.variable_scope("hidden"):
                    dec_hidden_enc = tf.nn.relu(linear(tf.concat(axis=1, values=(zy_phi, m)), self.n_dec_hidden))
                with tf.variable_scope("mu"):
                    dec_mu = linear(dec_hidden_enc, self.n_x)
                with tf.variable_scope("sigma"):
                    dec_sigma = tf.nn.softplus(linear(dec_hidden_enc, self.n_x))
                with tf.variable_scope("rho"):
                    dec_rho = tf.nn.sigmoid(linear(dec_hidden_enc, self.n_x))

            eps2 = tf.random_normal((tf.shape(x)[0], self.n_x), 0.0, 1.0, dtype=tf.float32)
            dec_x = tf.add(dec_mu, tf.multiply(dec_sigma, eps2))

            output, state_update = self.lstm(tf.concat(axis=1, values=(xy_phi, zy_phi)), LSTMStateTuple(c, m))
            # TODO: recheck it
        # return tf.nn.rnn_cell.LSTMStateTuple(h=(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma), c=state2)
        cell_output = tf.concat(values=(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_x,
                                        prior_mu, prior_sigma, z_encoder),
                                axis=1)
        c_update, m_update = state_update
        tpp_cvrnn_state = TPPCVRNNStateTuple(c=c_update, h=m_update, mu_p=prior_mu, sigma_p=prior_sigma)
        return cell_output, tpp_cvrnn_state


class TPPCVRNN():
    def __init__(self, config):
        self.config = config
        self.target_data_ph = tf.placeholder(dtype=tf.float32,
                                             shape=[None, self.config.Learn.max_seq_length,
                                                    self.config.Arch.CVRNN.x_dim], name='target_data')
        self.input_data_ph = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.config.Learn.max_seq_length,
                                                   self.config.Arch.CVRNN.x_dim +
                                                   self.config.Arch.CVRNN.y_dim +
                                                   self.config.Arch.CVRNN.latent_dim +
                                                   self.config.Arch.CVRNN.latent_dim +
                                                   1],
                                            name='input_data')

        self.selection_matrix_ph = tf.placeholder(dtype=tf.int32,
                                                  shape=[None, self.config.Learn.max_seq_length],
                                                  name='selection_matrix')
        self.sarsa_target_ph = tf.placeholder(dtype=tf.float32,
                                              shape=[None, 3], name='sarsa_target')

        self.trace_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='trace_length')
        self.cell = None
        self.initial_state_c = None
        self.initial_state_h = None
        self.kl_loss = None
        self.likelihood_loss = None
        # self.cost = None
        self.train_ll_op = None
        self.final_state_c = None
        self.final_state_h = None
        self.enc_mu = None
        self.enc_sigma = None
        self.dec_mu = None
        self.dec_sigma = None
        self.dec_x = None
        self.prior_mu = None
        self.prior_sigma = None
        self.output = None
        self.train_general_op = None
        self.train_td_op = None
        self.sarsa_output = None
        self.td_loss = None
        self.td_avg_diff = None
        self.deterministic_decoder = True
        self.cell_output_dim_list = [self.config.Arch.CVRNN.latent_dim, self.config.Arch.CVRNN.latent_dim,
                                     self.config.Arch.CVRNN.x_dim, self.config.Arch.CVRNN.x_dim,
                                     self.config.Arch.CVRNN.x_dim, self.config.Arch.CVRNN.latent_dim,
                                     self.config.Arch.CVRNN.latent_dim, self.config.Arch.CVRNN.latent_dim]
        self.cell_output_names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma",
                                  "dec_x", "prior_mu", "prior_sigma", "z_encoder"]

        self.sarsa_lstm_cell = []
        self.build_sarsa()

    def build_sarsa(self):
        with tf.name_scope("sarsa"):
            with tf.name_scope("LSTM-layer"):
                for i in range(self.config.Arch.SARSA.lstm_layer_num):
                    self.sarsa_lstm_cell.append(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.config.Arch.SARSA.h_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.05, 0.05)))

    def __call__(self):

        def tf_normal(target_x, mu, s, rho):  # TODO: bug, bug, bug, dynamic_rnn has zero-out, but anyway we ignore it
            with tf.variable_scope('normal'):
                ss = tf.maximum(1e-10, tf.square(s))
                # norm = tf.subtract(y[:, :args.chunk_samples], mu)  # TODO: why?
                norm = tf.subtract(target_x, mu)
                z = tf.div(tf.square(norm), ss)
                denom_log = tf.log(2 * np.pi * ss, name='denom_log')
                result = tf.reduce_sum(z + denom_log, 1) / 2  # -
                # (tf.log(tf.maximum(1e-20,rho),name='log_rho')*(1+y[:,args.chunk_samples:])
                # +tf.log(tf.maximum(1e-20,1-rho),name='log_rho_inv')*(1-y[:,args.chunk_samples:]))/2, 1)
            return result

        def tf_td_loss(sarsa_output, sarsa_target_ph, condition, if_last_output):
            with tf.variable_scope('n2_loss'):
                td_loss_all = tf.reduce_mean(tf.square(sarsa_output - sarsa_target_ph), axis=-1)
                zero_loss_all = tf.zeros(shape=[tf.shape(td_loss_all)[0]])
                if if_last_output:
                    return td_loss_all
                else:
                    return tf.where(condition=condition, x=td_loss_all, y=zero_loss_all)

        def tf_td_diff(sarsa_output, sarsa_target_ph, condition, if_last_output):
            with tf.variable_scope('n1_loss'):
                td_loss_all = tf.reduce_mean(tf.abs(sarsa_output - sarsa_target_ph), axis=-1)
                zero_loss_all = tf.zeros(shape=[tf.shape(td_loss_all)[0]])
                if if_last_output:
                    return td_loss_all
                else:
                    return tf.where(condition=condition, x=td_loss_all, y=zero_loss_all)

        def tf_cross_entropy(target_x, dec_x, condition):
            with tf.variable_scope('cross_entropy'):
                ce_loss_all = tf.losses.softmax_cross_entropy(onehot_labels=target_x,
                                                              logits=dec_x, reduction=tf.losses.Reduction.NONE)

                zero_loss_all = tf.zeros(shape=[tf.shape(ce_loss_all)[0]])
                return tf.where(condition=condition, x=ce_loss_all, y=zero_loss_all)

        def tf_kl_gaussian(mu_1, sigma_1, mu_2, sigma_2, condition):
            with tf.variable_scope("kl_gaussian"):
                kl_loss_all = tf.reduce_sum(0.5 * (
                        2 * tf.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
                        - 2 * tf.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
                        + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9, (tf.square(sigma_2))) - 1
                ), 1)
                zero_loss_all = tf.zeros(shape=[tf.shape(kl_loss_all)[0]])

                return tf.where(condition=condition, x=kl_loss_all, y=zero_loss_all)

        def get_cvrnn_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_x, prior_mu, prior_sigma, target_x,
                               condition, deterministci_decoder):
            if deterministci_decoder:
                likelihood_loss = tf_cross_entropy(dec_x=dec_mu, target_x=target_x, condition=condition)
            else:
                likelihood_loss = tf_cross_entropy(dec_x=dec_x, target_x=target_x, condition=condition)
            kl_loss = tf_kl_gaussian(enc_mu, enc_sigma, prior_mu, prior_sigma, condition)
            # likelihood_loss = tf_normal(target_x, dec_mu, dec_sigma, dec_rho)

            # kl_loss = tf.zeros(shape=[tf.shape(kl_loss)[0]])  # TODO: why if we only optimize likelihood_loss
            return kl_loss, likelihood_loss

        def get_td_lossfunc(sarsa_output, sarsa_target_ph, condition, if_last_output):
            td_loss = tf_td_loss(sarsa_output, sarsa_target_ph, condition, if_last_output=if_last_output)
            td_diff = tf_td_diff(sarsa_output, sarsa_target_ph, condition, if_last_output=if_last_output)
            return td_loss, td_diff

        # self.args = args
        # if sample:
        #     args.batch_size = 1
        #     args.seq_length = 1
        batch_size = tf.shape(self.input_data_ph)[0]
        with tf.variable_scope('cvrnn'):

            self.cell = TemporalProcessCVRNNCell(x_dim=self.config.Arch.CVRNN.x_dim, y_dim=self.config.Arch.CVRNN.y_dim,
                                                 h_dim=self.config.Arch.CVRNN.hidden_dim,
                                                 z_dim=self.config.Arch.CVRNN.latent_dim,
                                                 output_dim_list=self.cell_output_dim_list)

            self.initial_state_c, self.initial_state_h, self.initial_state_prior_mu, self.initial_state_prior_sigma \
                = self.cell.zero_state(batch_size=tf.shape(self.input_data_ph)[0],
                                       dtype=tf.float32)

            tpp_cvrnn_initial_state = TPPCVRNNStateTuple(
                self.initial_state_c, self.initial_state_h, self.initial_state_prior_mu, self.initial_state_prior_sigma
            )

            flat_target_data = tf.reshape(self.target_data_ph, [-1, self.config.Arch.CVRNN.x_dim])
            tppcvrnn_rnn_outputs, last_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.input_data_ph,
                                                                 sequence_length=self.trace_length_ph,
                                                                 initial_state=tpp_cvrnn_initial_state)
        # print outputs
        # outputs = map(tf.pack,zip(*outputs))
        tppcvrnn_rnn_outputs = tf.split(value=tf.transpose(a=tppcvrnn_rnn_outputs, perm=[1, 0, 2]),
                                        num_or_size_splits=[1] * self.config.Learn.max_seq_length, axis=0)
        outputs_reshape = []
        outputs_all = []
        for output in tppcvrnn_rnn_outputs:
            output = tf.squeeze(output, axis=0)
            output = tf.split(value=output, num_or_size_splits=self.cell.output_dim_list, axis=1)
            outputs_all.append(output)

        for n, name in enumerate(self.cell_output_names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs_all])
                x = tf.transpose(x, [1, 0, 2])
                x = tf.reshape(x, [batch_size * self.config.Learn.max_seq_length, self.cell_output_dim_list[n]])
                outputs_reshape.append(x)

        [self.enc_mu, self.enc_sigma, self.dec_mu, self.dec_sigma,
         self.dec_x, self.prior_mu, self.prior_sigma, z_encoder] = outputs_reshape
        self.z_encoder = z_encoder
        self.final_state_c, self.final_state_h, self.final_prior_mu, self.final_prior_sigma = last_state

        condition = tf.cast(tf.reshape(self.selection_matrix_ph,
                                       shape=[tf.shape(self.selection_matrix_ph)[0] *
                                              tf.shape(self.selection_matrix_ph)[1]]), tf.bool)

        # zero_output_all = tf.zeros(shape=[tf.shape(self.dec_x)[0]])
        # self.output = tf.where(condition=condition, x=self.dec_x, y=zero_output_all)
        if self.deterministic_decoder:
            decoder_output = self.dec_mu
        else:
            decoder_output = self.dec_x
        self.output = tf.reshape(tf.nn.softmax(decoder_output),
                                 shape=[batch_size, tf.shape(self.input_data_ph)[1], -1])

        kl_loss, likelihood_loss = get_cvrnn_lossfunc(self.enc_mu, self.enc_sigma, self.dec_mu, self.dec_sigma,
                                                      self.dec_x, self.prior_mu,
                                                      self.prior_sigma, flat_target_data, condition,
                                                      self.deterministic_decoder)

        with tf.variable_scope('cvrnn_cost'):
            self.kl_loss = tf.reshape(kl_loss, shape=[batch_size, self.config.Learn.max_seq_length, -1])
            self.likelihood_loss = tf.reshape(likelihood_loss, shape=[batch_size, self.config.Learn.max_seq_length, -1])

        tvars_cvrnn = tf.trainable_variables(scope='cvrnn')
        for t in tvars_cvrnn:
            print ('cvrnn_var: ' + str(t.name))
        cvrnn_grads = tf.gradients(tf.reduce_mean(self.likelihood_loss + self.kl_loss), tvars_cvrnn)
        ll_grads = tf.gradients(tf.reduce_mean(self.likelihood_loss), tvars_cvrnn)
        # grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.config.Learn.learning_rate)
        self.train_ll_op = optimizer.apply_gradients(zip(ll_grads, tvars_cvrnn))
        self.train_general_op = optimizer.apply_gradients(zip(cvrnn_grads, tvars_cvrnn))
        # self.saver = tf.train.Saver(tf.all_variables())

        with tf.variable_scope('sarsa'):

            data_input_sarsa = self.input_data_ph[
                               :, :,
                               self.config.Arch.CVRNN.x_dim:self.config.Arch.CVRNN.y_dim + self.config.Arch.CVRNN.x_dim]

            self.select_index = tf.range(0, batch_size) * self.config.Learn.max_seq_length + (self.trace_length_ph - 1)
            z_encoder_sarsa = tf.reshape(self.z_encoder, shape=[batch_size, self.config.Learn.max_seq_length,
                                                                self.config.Arch.CVRNN.latent_dim])

            # z_encoder_last = tf.gather(z_encoder, self.select_index)
            # self.z_encoder_last = z_encoder_last
            # sarsa_y_last = tf.gather(data_input_sarsa, self.select_index)
            # self.sarsa_y_last = sarsa_y_last

            for i in range(self.config.Arch.SARSA.lstm_layer_num):
                rnn_output = None
                for i in range(self.config.Arch.SARSA.lstm_layer_num):
                    rnn_input = tf.concat([data_input_sarsa, z_encoder_sarsa], axis=2) if i == 0 else rnn_output
                    rnn_output, rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                        inputs=rnn_input, cell=self.sarsa_lstm_cell[i],
                        sequence_length=self.trace_length_ph, dtype=tf.float32,
                        scope='sarsa_rnn_{0}'.format(str(i)))
                tppcvrnn_rnn_outputs = tf.stack(rnn_output)
                # Indexing
                rnn_last = tf.gather(tf.reshape(tppcvrnn_rnn_outputs,
                                                [-1, self.config.Arch.SARSA.h_size]), self.select_index)

            for j in range(self.config.Arch.SARSA.dense_layer_number - 1):
                sarsa_input = rnn_last if j == 0 else sarsa_output
                sarsa_output = tf.nn.relu(linear(sarsa_input, output_size=self.config.Arch.SARSA.dense_layer_size,
                                                 scope='dense_Linear'))
            sarsa_input = rnn_last if self.config.Arch.SARSA.dense_layer_number == 1 else sarsa_output
            sarsa_output = linear(sarsa_input, output_size=3, scope='output_Linear')
            self.sarsa_output = tf.nn.softmax(sarsa_output)

        with tf.variable_scope('td_cost'):
            td_loss, td_diff = get_td_lossfunc(self.sarsa_output, self.sarsa_target_ph, condition, if_last_output=True)
            # self.td_loss = tf.reshape(td_loss, shape=[tf.shape(self.input_data_ph)[0],
            #                                           self.config.Learn.max_seq_length, -1])
            self.td_loss = td_loss
            self.td_avg_diff = tf.reduce_mean(td_diff)

        tvars_td = tf.trainable_variables(scope='sarsa')
        for t in tvars_td:
            print ('td_var: ' + str(t.name))
        td_grads = tf.gradients(tf.reduce_mean(self.td_loss), tvars_td)
        self.train_td_op = optimizer.apply_gradients(zip(td_grads, tvars_td))



if __name__ == '__main__':
    tpp_cvrnn_config_path = "../environment_settings/icehockey_cvrnn_config.yaml"
    cvrnn_config = TPPCVRNNConfig.load(tpp_cvrnn_config_path)
    cvrnn = TPPCVRNN(config=cvrnn_config)
    cvrnn()
