from nn_structure.lstm_prediction_nn import Td_Prediction_NN
from config.lstm_prediction_config import LSTMCongfig
import tensorflow as tf


def run():
    tt_lstm_config_path = "../ice_hockey_prediction.yaml"
    lstm_prediction_config = LSTMCongfig.load(tt_lstm_config_path)

    sess = tf.Session()
    model = Td_Prediction_NN(
        feature_number=lstm_prediction_config.Learn.feature_number,
        h_size=lstm_prediction_config.Arch.LSTM.h_size,
        max_trace_length=lstm_prediction_config.Learn.max_seq_length,
        learning_rate=lstm_prediction_config.Learn.learning_rate,
        embed_size=lstm_prediction_config.Learn.embed_size,
        output_layer_size=lstm_prediction_config.Learn.output_layer_size,
        lstm_layer_num=lstm_prediction_config.Arch.LSTM.lstm_layer_num,
        dense_layer_num=lstm_prediction_config.Learn.dense_layer_num,
        apply_softmax=lstm_prediction_config.Learn.apply_softmax)
    model.initialize_ph()
    model.build()
    model.call()
    # train_network(sess=sess, model=model, )
    sess.close()


if __name__ == '__main__':
    run()
