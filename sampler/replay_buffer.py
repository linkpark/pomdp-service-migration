import numpy as np
import random



class SequentialReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, paths):
        for path in paths:
            ob = path["observations"]
            action = path["actions"]
            reward = path["un_norm_rewards"]
            max_q_value = path["max_q_value"]
            max_q_value = np.append(max_q_value, 0)
            next_max_q_value = max_q_value[1:]

            data = (ob, action, reward, next_max_q_value)

            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, actions, rewards, next_max_q_values = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            ob, action, reward, next_max_q_value = data
            obses.append(np.array(ob, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_max_q_values.append(next_max_q_value)

        return np.array(obses), np.array(actions), np.array(rewards), np.array(next_max_q_values)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def size(self):
        return len(self._storage)


if __name__ == "__main__":
    from environment.batch_migration_env import EnvironmentParameters
    from environment.batch_migration_env import BatchMigrationEnv
    from policies.rnn_q_network import RNNQNetwork
    from sampler.migration_sampler import MigrationSamplerForDRQN

    possion_rate_vector = [7, 10, 8, 14, 15, 6, 20, 18, 11, 17, 20, 9, 8, 14, 9, 15, 8, 17, 9, 9, 10, 7, 17, 10,
                           13, 12, 5, 8, 10, 13, 19, 15, 10, 9, 10, 18, 12, 13, 5, 11, 7, 8, 8, 19, 15, 15, 6, 10,
                           5, 20, 17, 5, 5, 16, 5, 19, 19, 19, 9, 20, 17, 14, 17, 17]

    # start point (41.856, 12.442), end point (41.928,12.5387), a region in Roman, Italy.
    env_default_parameters = EnvironmentParameters(trace_start_index=0,
                                                   num_traces=10,
                                                   num_base_station=64, optical_fiber_trans_rate=60.0,
                                                   server_poisson_rate=possion_rate_vector, client_poisson_rate=4,
                                                   server_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                   server_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                   ratio_lower_bound=100.0,
                                                   client_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                   client_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                   ratio_higher_bound=3200.0, map_width=8000.0, map_height=8000.0,
                                                   num_horizon_servers=8, num_vertical_servers=8,
                                                   traces_file_path='../environment/rome_traces_coordinate.txt',
                                                   transmission_rates=[20.0, 16.0, 12.0, 8.0, 4.0],
                                                   trace_length=100,
                                                   trace_interval=10,
                                                   is_full_observation=False,
                                                   is_full_action=True)

    env = BatchMigrationEnv(env_default_parameters)

    q_network = RNNQNetwork(observation_dim=env._state_dim,
                            action_dim=env._action_dim,
                            rnn_parameter=256,
                            fc_parameters=128,
                            epsilon=0.1)

    obs = env.reset()
    initial_state = q_network.get_initial_hidden_state(obs)

    sampler = MigrationSamplerForDRQN(env,
                               policy=q_network,
                               batch_size=100,
                               num_environment_per_core=2,
                               max_path_length=100,
                               parallel=True,
                               num_process=5,
                               is_norm_reward=True)

    paths = sampler.obtain_samples()

    print("paths length: ", len(paths))
    print("q_values_shape: ", paths[0]['max_q_value'].shape)

    replay_buffer = SequentialReplayBuffer(size=1000)
    replay_buffer.add(paths)

    obs, actions, rewards, q_values = replay_buffer.sample(200)

    print(obs.shape)
    print(actions.shape)
    print(rewards.shape)
    print("next q_value last value: ", q_values[:,-1])