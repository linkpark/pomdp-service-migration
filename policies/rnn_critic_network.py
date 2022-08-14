import tensorflow as tf
import numpy as np

class RNNValueNetwork(tf.keras.Model):
    def __init__(self,
                 observation_dim,
                 rnn_parameters):
        super(RNNValueNetwork, self).__init__()
        self.observation_dim = observation_dim

        self.lstm_layer = tf.keras.layers.LSTM(units=rnn_parameters,
                                               return_sequences=True, return_state=True)

        self.projection_layer = tf.keras.layers.Dense(units=1)

    def call(self, observations):
        whole_seq_output, final_memory_state, final_carry_state = \
            self.lstm_layer(observations)

        values = tf.squeeze(self.projection_layer(whole_seq_output))
        return values

    def predict(self, observations):
        if len(observations.shape) == 2:
            observations = np.expand_dims(observations, axis=0)

        x = tf.convert_to_tensor(observations)

        whole_seq_output, final_memory_state, final_carry_state = \
            self.lstm_layer(x)

        values = tf.squeeze(self.projection_layer(whole_seq_output))
        return values