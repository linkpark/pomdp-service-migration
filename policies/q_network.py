import tensorflow as tf
import random
import numpy as np

class QNetwork(tf.keras.Model):
    def __init__(self,
                 epsilon,
                 observation_dim,
                 action_dim,
                 hidden_parameter,
                 fc_parameters,
                 is_decade_epsilons=True):
        super(QNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.epsilon = epsilon

        self.observation_projection_layer = tf.keras.layers.Dense(units=fc_parameters, activation="relu")

        self.hidden_layer = tf.keras.layers.Dense(
            units=hidden_parameter, activation="relu"
        )

        self.value_project_layer = tf.keras.layers.Dense(units=action_dim)

    def sample(self, observations):
        observations = tf.convert_to_tensor(observations)
        batch_size = observations.shape[0]
        x = self.observation_projection_layer(observations)

        x = self.hidden_layer(x)

        logits = self.value_project_layer(x)

        random_number = random.random()
        if random_number < self.epsilon:
            action = tf.random.uniform(shape=[batch_size], maxval=self.action_dim, dtype=tf.int32).numpy()
        else:
            action = tf.math.argmax(logits, axis=-1)
            action = tf.squeeze(action).numpy()

        return action

    def get_max_q_value(self, observations):
        observations = tf.convert_to_tensor(observations)

        x = self.observation_projection_layer(observations)

        x = self.hidden_layer(x)
        # logits = tf.expand_dims(self.projection_layer(x), axis=1)
        q_values = self.value_project_layer(x)

        max_q_values = tf.math.reduce_max(q_values, axis=-1).numpy()

        return max_q_values


    def greedy_sample(self, observations):
        observations = tf.convert_to_tensor(observations)
        x = self.observation_projection_layer(observations)

        x = self.hidden_layer(x)
        q_values = self.value_project_layer(x)

        action = tf.math.argmax(q_values, axis=-1)
        action = tf.squeeze(action).numpy()

        return action


    def call(self, observations):
        ob_t = tf.convert_to_tensor(observations)
        x = self.observation_projection_layer(ob_t)

        loggits = self.hidden_layer(x)

        q_values = self.value_project_layer(loggits)

        return q_values

