import tensorflow as tf
import tensorflow_probability as tfp
from policies.distributions.categorical_pd import CategoricalPd
import numpy as np
import random

class RNNPolicy(tf.keras.Model):
    def __init__(self,
                 observation_dim,
                 action_dim,
                 rnn_parameter,
                 fc_parameters):
        super(RNNPolicy, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.observation_projection_layer = tf.keras.layers.Dense(units=fc_parameters, activation="relu")
        self.action_projection_layer = tf.keras.layers.Dense(units=action_dim, activation="relu")

        self.lstm_cell = tf.keras.layers.LSTMCell(
            units=rnn_parameter
        )
        self.training_rnn = tf.keras.layers.RNN(cell=self.lstm_cell,
                                                return_sequences=True,
                                                return_state=True)

        self.projection_layer = tf.keras.layers.Dense(units=action_dim)
        self.distribution = CategoricalPd(action_dim)

    # Notice Tensorflow v2.0 GRU has bugs.
    def sample(self, observations, actions, hidden_state):
        observations = tf.convert_to_tensor(observations)
        actions = tf.expand_dims(tf.convert_to_tensor(actions), axis=-1)

        actions = self.action_projection_layer(actions)
        observations = self.observation_projection_layer(observations)

        x = tf.concat([observations, actions], axis=-1)
        x, hidden_state = self.lstm_cell(x, hidden_state)
        logits = tf.expand_dims(self.projection_layer(x), axis=1)

        predicted_sampler = tfp.distributions.Categorical(logits=logits)
        action = tf.squeeze(predicted_sampler.sample(seed=random.seed())).numpy()

        return action, hidden_state

    def greedy_sample(self, observations, actions, hidden_state):
        observations = tf.convert_to_tensor(observations)
        actions = tf.expand_dims(tf.convert_to_tensor(actions), axis=-1)

        actions = self.action_projection_layer(actions)
        observations = self.observation_projection_layer(observations)

        x = tf.concat([observations, actions], axis=-1)
        x, hidden_state = self.lstm_cell(x, hidden_state)
        logits = tf.expand_dims(self.projection_layer(x), axis=1)

        action = tf.math.argmax(logits, axis=-1)

        return action, hidden_state

    def sequence_sample(self, observations):
        observations = tf.convert_to_tensor(observations)
        batch_size = observations.shape[0]
        time_steps = observations.shape[1]

        hidden_state = self.lstm_cell.get_initial_state(inputs=observations)
        actions = []
        seq_logits = []
        action = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
        for i in range(time_steps):
            ob_t = observations[:,i,:]
            action = self.action_projection_layer(action)
            ob_t = self.observation_projection_layer(ob_t)

            x = tf.concat([ob_t, action], axis=-1)
            x, hidden_state = self.lstm_cell(x, hidden_state)
            logits = tf.expand_dims(self.projection_layer(x), axis=1)

            seq_logits.append(logits)

            predicted_sampler = tfp.distributions.Categorical(logits=logits)
            action = predicted_sampler.sample(seed=random.seed())
            actions.append(action)

        ret_actions = tf.concat(actions, axis=1)
        ret_logtis = tf.concat(seq_logits, axis=1)

        return ret_actions, ret_logtis

    def get_initial_hidden_state(self, observations):
        observations = tf.convert_to_tensor(observations)
        hidden_state = self.lstm_cell.get_initial_state(inputs=observations)

        return hidden_state

    def call(self, observations, actions):
        ob_t = tf.convert_to_tensor(observations)
        actions = tf.expand_dims(tf.convert_to_tensor(actions, dtype=tf.float32), axis=-1)

        action = self.action_projection_layer(actions)
        ob_t = self.observation_projection_layer(ob_t)

        x = tf.concat([ob_t, action], axis=-1)

        whole_seq_output, _, _ = self.training_rnn(x)
        logits = self.projection_layer(whole_seq_output)

        pi = tf.nn.softmax(logits)
        actions = tf.math.argmax(logits)

        return pi, logits, actions

class RNNValueNet(tf.keras.Model):
    def __init__(self,
                 observation_dim,
                 action_dim,
                 rnn_parameter,
                 fc_parameters):
        super(RNNValueNet, self).__init__()
        self.observation_dim = observation_dim

        self.observation_projection_layer = tf.keras.layers.Dense(units=fc_parameters, activation="relu")
        self.action_projection_layer = tf.keras.layers.Dense(units=64, activation="relu")

        self.lstm_cell = tf.keras.layers.LSTMCell(
            units=rnn_parameter
        )
        self.training_rnn = tf.keras.layers.RNN(cell=self.lstm_cell,
                                                return_sequences=True,
                                                return_state=True)

        self.projection_layer = tf.keras.layers.Dense(units=1)
        self.distribution = CategoricalPd(action_dim)

    def predict(self, x):
        observations, actions = x
        ob_t = tf.convert_to_tensor(observations)
        actions = tf.expand_dims(tf.convert_to_tensor(actions, dtype=tf.float32), axis=-1)

        action = self.action_projection_layer(actions)
        ob_t = self.observation_projection_layer(ob_t)

        x = tf.concat([ob_t, action], axis=-1)

        whole_seq_output, _, _ = self.training_rnn(x)


        values = tf.squeeze(self.projection_layer(whole_seq_output))

        return values

    def call(self, x):
        observations, actions = x
        ob_t = tf.convert_to_tensor(observations)
        actions = tf.expand_dims(tf.convert_to_tensor(actions, dtype=tf.float32), axis=-1)

        action = self.action_projection_layer(actions)
        ob_t = self.observation_projection_layer(ob_t)

        x = tf.concat([ob_t, action], axis=-1)

        whole_seq_output, _, _ = self.training_rnn(x)
        values = tf.squeeze(self.projection_layer(whole_seq_output))

        return values

class RNNPolicyWithValue(tf.keras.Model):
    def __init__(self,
                 observation_dim,
                 action_dim,
                 rnn_parameter,
                 embbeding_size):
        super(RNNPolicyWithValue, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=action_dim, output_dim=embbeding_size)

        self.observation_projection_layer = tf.keras.layers.Dense(units=128, activation="relu")
        #self.action_projection_layer = tf.keras.layers.Dense(units=fc_parameters, activation="relu")

        self.lstm_cell = tf.keras.layers.LSTMCell(
            units=rnn_parameter
        )

        self.training_rnn = tf.keras.layers.RNN(cell=self.lstm_cell,
                                                return_sequences=True,
                                                return_state=True)

        self.projection_layer = tf.keras.layers.Dense(units=action_dim)
        self.distribution = CategoricalPd(action_dim)
        self.value_fc_layer = tf.keras.layers.Dense(units=128, activation="relu")
        self.value_project_layer = tf.keras.layers.Dense(units=action_dim)

    def get_initial_hidden_state(self, observations):
        observations = tf.convert_to_tensor(observations)
        hidden_state = self.lstm_cell.get_initial_state(inputs=observations)

        return hidden_state

    # The input of the sample function is 2-D (batch_size, ob_dim)
    def sample(self, observations, actions, hidden_state):
        actions = np.array(actions, dtype=np.int32)
        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        # modify observations
        user_position = tf.cast(observations[:,0], dtype=tf.int32)
        user_position_embeddings = self.embedding_layer(user_position)

        observations = tf.concat((user_position_embeddings, observations[:,1:]), axis=-1)

        actions = self.embedding_layer(actions)
        observations = self.observation_projection_layer(observations)

        x = tf.concat([observations, actions], axis=-1)
        x, hidden_state = self.lstm_cell(x, hidden_state)
        #logits = tf.expand_dims(self.projection_layer(x), axis=1)
        logits = self.projection_layer(x)

        predicted_sampler = tfp.distributions.Categorical(logits=logits)
        action = tf.squeeze(predicted_sampler.sample(seed=random.seed())).numpy()

        return action, hidden_state

    def greedy_sample(self, observations, actions, hidden_state):
        actions = np.array(actions, dtype=np.int32)
        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        # modify observations
        user_position = tf.cast(observations[:,0], dtype=tf.int32)
        user_position_embeddings = self.embedding_layer(user_position)

        observations = tf.concat((user_position_embeddings, observations[:, 1:]), axis=-1)

        actions = self.embedding_layer(actions)
        observations = self.observation_projection_layer(observations)

        x = tf.concat([observations, actions], axis=-1)
        x, hidden_state = self.lstm_cell(x, hidden_state)
        logits = self.projection_layer(x)

        action = tf.math.argmax(logits, axis=-1)
        #action = tf.squeeze(action).numpy()
        action = action.numpy()
        return action, hidden_state

    def predict(self, x):
        observations, actions = x
        actions = np.array(actions, dtype=np.int32)
        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        action = self.embedding_layer(actions)

        user_position = tf.cast(observations[:,:, 0], dtype=tf.int32)
        user_position_embeddings = self.embedding_layer(user_position)

        observations = tf.concat((user_position_embeddings, observations[:,:, 1:]), axis=-1)
        observations = self.observation_projection_layer(observations)

        x = tf.concat([observations, action], axis=-1)

        whole_seq_output, _, _ = self.training_rnn(x)

        logits = self.projection_layer(whole_seq_output)

        pi = tf.nn.softmax(logits)

        q_values_logits =  self.value_fc_layer(whole_seq_output)
        q_values = self.value_project_layer(q_values_logits)


        values = tf.reduce_sum( (pi * q_values), axis=-1)

        return values

    def call(self, observations, actions):
        actions = np.array(actions, dtype=np.int32)
        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        action = self.embedding_layer(actions)

        user_position = tf.cast(observations[:, :, 0],dtype=tf.int32)
        user_position_embeddings = self.embedding_layer(user_position)

        observations = tf.concat((user_position_embeddings, observations[:, :, 1:]), axis=-1)
        observations = self.observation_projection_layer(observations)

        x = tf.concat([observations, action], axis=-1)

        whole_seq_output, _, _ = self.training_rnn(x)
        logits = self.projection_layer(whole_seq_output)

        q_values_logits = self.value_fc_layer(whole_seq_output)
        q_values = self.value_project_layer(q_values_logits)

        pi = tf.nn.softmax(logits)
        actions = tf.math.argmax(logits)
        values = tf.reduce_sum( (pi * q_values), axis=-1)

        return pi, logits, actions

if __name__ == "__main__":
    # test the fc policy:
    from environment.migration_env import EnvironmentParameters
    from environment.migration_env import MigrationEnv

    import numpy as np

    possion_rate_vector = np.random.randint(5, 21, size=63)
    print("possion_rate_vector is: ", possion_rate_vector)

    env_default_parameters = EnvironmentParameters(trace_start_index=0,
                                                   num_traces=10,
                                                   num_base_station=63, optical_fiber_trans_rate=60.0,
                                                   server_poisson_rate=possion_rate_vector, client_poisson_rate=4,
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
                                                   trace_interval=5,
                                                   is_full_observation=False,
                                                   is_full_action=True)

    env = MigrationEnv(env_default_parameters)
    print("action spec: ", env.action_spec())
    print("observes spec: ", env.observation_spec())

    rnn_policy = RNNPolicyWithValue(observation_dim=env._state_dim,
                 action_dim=env._action_dim,
                 rnn_parameter=128,
                 embbeding_size=2)

    obs = np.array([env.reset(), env.reset(),env.reset(),env.reset()])
    #obs = np.array([obs,obs,obs,obs,obs,obs,obs]).swapaxes(0,1)

    hidden_state = rnn_policy.get_initial_hidden_state(obs)
    actions = np.zeros(shape=(obs.shape[0],), dtype=np.float32)

    actions, ret_logtis = rnn_policy.sample(obs, actions, hidden_state)
    print("actions: ", actions)
    for action in actions:
        env.step(action)

    obs = np.expand_dims(obs, axis=1)
    actions = np.expand_dims(actions, axis=1)
    values = rnn_policy.predict((obs, actions))

    print("values shape", values.shape)

    actions = actions
    shift_actions = np.column_stack(
        (np.zeros(actions.shape[0], dtype=np.int32), actions[:, 0:-1]))


    print("actions is : ", actions)
    print("shift_actions is: ", shift_actions)

    print("action shape: ", actions.shape)
    print("logits shape: ", np.array(ret_logtis).shape)
    print("logits :",ret_logtis[0][0][0])

    pi, logits, _, _ = rnn_policy(obs, shift_actions)
    # print("policy is: ", pi)
    print("logits: ", logits[0][0][0])
    #print("observations shape is: ", obs.shape)
    #actions = rnn_policy.sample(obs)

    # Test policy sample
    obs = np.array([env.reset(),env.reset(),env.reset(),env.reset()])
    initial_state = rnn_policy.get_initial_hidden_state(obs)
    actions = np.array([0, 0, 0,0])
    sample_action, hidden_state = rnn_policy.sample(obs, actions, initial_state)

    print("sampled action: ", sample_action)
    #print("hidden state: ", hidden_state)