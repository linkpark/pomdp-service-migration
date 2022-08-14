import tensorflow as tf
import tensorflow_probability as tfp
from policies.distributions.categorical_pd import CategoricalPd

import random

class FCCategoricalPolicy(tf.keras.Model):
    def __init__(self,
                 observation_dim,
                 action_dim,
                 fc_parameters):
        super(FCCategoricalPolicy, self).__init__()
        self.observation_dim = observation_dim
        self.fc_layers = []

        for parameter in fc_parameters:
            self.fc_layers.append(tf.keras.layers.Dense(units=parameter, activation='relu'))

        self.projection_layer = tf.keras.layers.Dense(units=action_dim)
        # self.value_projection_layer = tf.keras.layers.Dense(units=action_dim)

        self.distribution = CategoricalPd(action_dim)

    def sample(self, observations):
        x = tf.convert_to_tensor(observations)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        logits = self.projection_layer(x)

        predicted_sampler = tfp.distributions.Categorical(logits=logits)
        actions = predicted_sampler.sample(seed=random.seed())

        return actions

    def call(self, observations):
        x = observations
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        logits = self.projection_layer(x)
        # q_values = self.value_projection_layer(x)
        #
        pi = tf.nn.softmax(logits)
        actions = tf.math.argmax(logits)
        # values =  tf.math.reduce_sum(pi * q_values, axis=-1)

        return pi, logits, actions

    # def predict(self, observations):
    #     x = observations
    #     for fc_layer in self.fc_layers:
    #         x = fc_layer(x)
    #
    #     logits = self.projection_layer(x)
    #     q_values = self.value_projection_layer(x)
    #
    #     pi = tf.nn.softmax(logits)
    #     values  = tf.math.reduce_sum(pi * q_values, axis=-1)
    #
    #     return values

    def step(self, observations):
        x = observations
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        logits = self.projection_layer(x)
        actions = tf.math.argmax(logits)

        return actions

class FCValueNetwork(tf.keras.Model):
    def __init__(self,
                 observation_dim,
                 fc_parameters):
        super(FCValueNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.fc_layers = []

        for parameter in fc_parameters:
            self.fc_layers.append(tf.keras.layers.Dense(units=parameter, activation='relu'))

        self.projection_layer = tf.keras.layers.Dense(units=1)

    def call(self, observations):
        x = observations
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        values = tf.squeeze(self.projection_layer(x))
        return values

    def predict(self, observations):
        x = tf.convert_to_tensor(observations)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        values = tf.squeeze(self.projection_layer(x))
        return values

class FCCategoricalPolicyWithValue(tf.keras.Model):
    def __init__(self,
                 observation_dim,
                 action_dim,
                 fc_parameters):
        super(FCCategoricalPolicyWithValue, self).__init__()
        self.observation_dim = observation_dim
        self.fc_layers = []

        for parameter in fc_parameters:
            self.fc_layers.append(tf.keras.layers.Dense(units=parameter, activation='relu'))

        self.projection_layer = tf.keras.layers.Dense(units=action_dim)
        self.value_projection_layer = tf.keras.layers.Dense(units=action_dim)

        self.distribution = CategoricalPd(action_dim)

    def sample(self, observations):
        x = tf.convert_to_tensor(observations)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        logits = self.projection_layer(x)

        predicted_sampler = tfp.distributions.Categorical(logits=logits)
        actions = predicted_sampler.sample(seed=random.seed())

        return actions

    def call(self, observations):
        x = observations
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        logits = self.projection_layer(x)
        q_values = self.value_projection_layer(x)
        #
        pi = tf.nn.softmax(logits)
        actions = tf.math.argmax(logits)
        values = tf.math.reduce_sum(pi * q_values, axis=-1)

        return pi, logits, values

    def predict(self, observations):
        x = tf.convert_to_tensor(observations)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        logits = self.projection_layer(x)
        q_values = self.value_projection_layer(x)

        pi = tf.nn.softmax(logits)
        values  = tf.math.reduce_sum(pi * q_values, axis=-1)

        return values

    def step(self, observations):
        x = observations
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        logits = self.projection_layer(x)
        actions = tf.math.argmax(logits)

        return actions

if __name__ == "__main__":
    # test the fc policy:
    from environment.migration_env import EnvironmentParameters
    from environment.migration_env import MigrationEnv

    import numpy as np

    env_default_parameters = EnvironmentParameters(num_base_station=63, optical_fiber_trans_rate=60.0,
                                                   server_poisson_rate=20, client_poisson_rate=4,
                                                   server_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                   server_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                   ratio_lower_bound=100.0,
                                                   client_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                   client_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                   ratio_higher_bound=3200.0, map_width=4500.0, map_height=3500.0,
                                                   num_horizon_servers=9, num_vertical_servers=7,
                                                   traces_file_path='../environment/default_scenario_LocationSnapshotReport.txt',
                                                   transmission_rates=[20.0, 16.0, 12.0, 8.0, 4.0],
                                                   trace_length=100,
                                                   is_full_observation=True,
                                                   is_full_action=True)

    env = MigrationEnv(env_default_parameters)
    print("action spec: ", env.action_spec())
    print("observes spec: ", env.observation_spec())

    fc_policy = FCCategoricalPolicy(observation_dim = 127,
                 action_dim=64,
                 fc_parameters=[64, 32])

    obs = np.array([env.reset()])
    obs = np.array([obs,obs]).swapaxes(0,1)

    print("observations shape is: ", obs.shape)

    actions, logits, _ = fc_policy(obs)
    values = fc_policy.predict(obs)

    print("actions", actions.numpy().shape)
    print("logits", logits.numpy().shape)
    print("values", values.numpy().shape)
    print()