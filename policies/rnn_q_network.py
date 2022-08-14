import tensorflow as tf
import random
import numpy as np

class RNNQNetwork(tf.keras.Model):
    def __init__(self,
                 epsilon,
                 observation_dim,
                 action_dim,
                 rnn_parameter,
                 fc_parameters,
                 is_decade_epsilons=True):
        super(RNNQNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.epsilon = epsilon

        self.observation_projection_layer = tf.keras.layers.Dense(units=fc_parameters, activation="relu")

        self.lstm_cell = tf.keras.layers.LSTMCell(
            units=rnn_parameter
        )
        self.training_rnn = tf.keras.layers.RNN(cell=self.lstm_cell,
                                                return_sequences=True,
                                                return_state=True)

        self.value_project_layer = tf.keras.layers.Dense(units=action_dim)

    def get_initial_hidden_state(self, observations):
        observations = tf.convert_to_tensor(observations)
        hidden_state = self.lstm_cell.get_initial_state(inputs=observations)

        return hidden_state

    def sample(self, observations, hidden_state):
        observations = tf.convert_to_tensor(observations)
        batch_size = observations.shape[0]
        x = self.observation_projection_layer(observations)

        x, hidden_state = self.lstm_cell(x, hidden_state)
        #logits = tf.expand_dims(self.projection_layer(x), axis=1)
        logits = self.value_project_layer(x)

        random_number = random.random()
        if random_number < self.epsilon:
            action = tf.random.uniform(shape=[batch_size], maxval=self.action_dim, dtype=tf.int32).numpy()
        else:
            action = tf.math.argmax(logits, axis=-1)
            action = tf.squeeze(action).numpy()

        return action, hidden_state

    def get_max_q_value(self, observations, hidden_state):
        observations = tf.convert_to_tensor(observations)
        batch_size = observations.shape[0]
        x = self.observation_projection_layer(observations)

        x, hidden_state = self.lstm_cell(x, hidden_state)
        # logits = tf.expand_dims(self.projection_layer(x), axis=1)
        q_values = self.value_project_layer(x)

        max_q_values = tf.math.reduce_max(q_values, axis=-1).numpy()

        return max_q_values


    def greedy_sample(self, observations, hidden_state):
        observations = tf.convert_to_tensor(observations)
        x = self.observation_projection_layer(observations)

        x, hidden_state = self.lstm_cell(x, hidden_state)
        q_values = self.value_project_layer(x)

        action = tf.math.argmax(q_values, axis=-1)
        action = tf.squeeze(action).numpy()

        return action, hidden_state


    def call(self, observations):
        ob_t = tf.convert_to_tensor(observations)
        x = self.observation_projection_layer(ob_t)

        whole_seq_output, _, _ = self.training_rnn(x)

        q_values = self.value_project_layer(whole_seq_output)

        return q_values
