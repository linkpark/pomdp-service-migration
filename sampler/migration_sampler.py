from sampler.sample_base import Sampler
from sampler.sample_base import SampleProcessor
from sampler.vectorized_env_executor import MetaParallelEnvExecutor
from sampler.vectorized_env_executor import MetaIterativeEnvExecutor
from sampler.vectorized_env_executor import ParallelEnvExecutor


from policies.random_migrate_policy import RandomMigratePolicy
from pyprind import ProgBar
from utils import logger
from utils import utils


import numpy as np
import time
import itertools

def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[],  env_infos=[],
                random_policy_rewards=[], un_norm_rewards=[], system_info=[], max_q_value=[])

def _get_empty_reward_paths_dict():
    return dict(un_norm_rewards=[])

class MigrationSamplerForDRQN(Sampler):
    def __init__(self,
                 env,
                 policy,
                 batch_size,
                 num_environment_per_core,
                 max_path_length,
                 parallel = True,
                 num_process = 6,
                 is_norm_reward = True,
                 is_rnn = True):
        super(MigrationSamplerForDRQN, self).__init__(env, policy, batch_size, max_path_length)
        self.total_timesteps_sampled = 0
        self.total_samples = batch_size * max_path_length
        self.num_of_environment = num_environment_per_core * num_process
        self.is_batch = env.is_batch
        self.env = env
        self.is_norm_reward = is_norm_reward
        self.is_rnn = is_rnn

        if parallel:
            self.vec_env = ParallelEnvExecutor(env, num_process,num_environment_per_core, max_path_length, is_batch_env=env.is_batch)
        else:
            self.vec_env = ParallelEnvExecutor(env, num_process, num_environment_per_core, max_path_length, is_batch_env=env.is_batch)

    def obtain_samples(self,
                    log=False,
                    log_prefix='',
                    reward_mean=0.0,
                    reward_std=1.0,
                    epsilon=0.0):
        paths = []

        n_samples = 0
        if self.is_batch:
            running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs * self.env._num_traces)]
        else:
            running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        self.policy.epsilon = epsilon
        # initial reset of envs
        obses = np.array(self.vec_env.reset(), dtype=np.float32)
        batch_size = obses.shape[0]

        if self.is_rnn:
            hidden_state = self.policy.get_initial_hidden_state(obses)

        while n_samples < self.total_samples:
            # step environments
            system_infos = self.vec_env.current_system_state()
            t = time.time()
            if self.is_rnn:
                actions, hidden_state = self.policy.sample(obses, hidden_state)
            else:
                actions = self.policy.sample(obses)

            policy_time = time.time() - t

            if self.is_rnn:
                max_q_values = self.policy.get_max_q_value(obses, hidden_state)
            else:
                max_q_values = self.policy.get_max_q_value(obses)

            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)

            env_time += time.time() - t

            new_samples = 0

            # when trajectory finish, initialize the hidden state
            if dones[0] and self.is_rnn:
                hidden_state = self.policy.get_initial_hidden_state(obses)

            for idx, observation, action, reward, max_q_value, env_info, done, system_info in zip(itertools.count(), obses, actions,
                                                                                    rewards, max_q_values, env_infos,
                                                                                    dones, system_infos):
                # append new samples to running paths
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["un_norm_rewards"].append(reward)
                running_paths[idx]["system_info"].append(system_info)
                running_paths[idx]["max_q_value"].append(max_q_value)

                # if running path is done, add it to paths and empty the running path
                if done:
                    if self.is_norm_reward:
                        rewards = utils.normalization(np.asarray(running_paths[idx]["un_norm_rewards"]))
                    else:
                        #rewards = (np.asarray(running_paths[idx]["un_norm_rewards"]) - reward_min) / (reward_max - reward_min)
                        #rewards = (np.asarray(running_paths[idx]["un_norm_rewards"]) - reward_mean) / (reward_std + 1e-8)
                        rewards = np.asarray(running_paths[idx]["un_norm_rewards"])
                    paths.append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        un_norm_rewards=np.asarray(running_paths[idx]["un_norm_rewards"]),
                        max_q_value = np.asarray(running_paths[idx]["max_q_value"]),
                        system_info = np.asarray(running_paths[idx]["system_info"]),
                        rewards=rewards
                    ))
                    new_samples += len(running_paths[idx]["un_norm_rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths


class MigrationSampler(Sampler):
    def __init__(self,
                 env,
                 policy,
                 batch_size,
                 num_environment_per_core,
                 max_path_length,
                 parallel = True,
                 num_process = 6,
                 is_norm_reward = True):
        super(MigrationSampler, self).__init__(env, policy, batch_size, max_path_length)
        self.total_timesteps_sampled = 0
        self.total_samples = batch_size * max_path_length
        self.num_of_environment = num_environment_per_core * num_process
        self.is_batch = env.is_batch
        self.env = env
        self.is_norm_reward = is_norm_reward

        if parallel:
            self.vec_env = ParallelEnvExecutor(env, num_process,num_environment_per_core, max_path_length, is_batch_env=env.is_batch)
        else:
            self.vec_env = ParallelEnvExecutor(env, num_process, num_environment_per_core, max_path_length, is_batch_env=env.is_batch)

    def obtain_samples(self,is_rnn=False,
                            log=False,
                            log_prefix='',
                            reward_mean=0.0,
                            reward_std=1.0):
        paths = []

        n_samples = 0
        if self.is_batch:
            running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs * self.env._num_traces)]
        else:
            running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        # initial reset of envs
        obses = np.array(self.vec_env.reset(), dtype=np.float32)

        if is_rnn:
            # initialize the action and hidden state, this is necessary due to the pomdp posterior.
            hidden_state = self.policy.get_initial_hidden_state(obses)
            actions = np.zeros(shape=(obses.shape[0],), dtype=np.float32)

        while n_samples < self.total_samples:
            # step environments
            system_infos = self.vec_env.current_system_state()
            t = time.time()
            if is_rnn:
                actions, hidden_state = self.policy.sample(obses, actions, hidden_state)
            else:
                actions = self.policy.sample(obses)

            policy_time = time.time() - t

            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)

            env_time += time.time() - t

            new_samples = 0

            # when trajectory finish, initialize the hidden state
            if dones[0] and is_rnn:
                hidden_state = self.policy.get_initial_hidden_state(obses)

            for idx, observation, action, reward, env_info, done, system_info in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos,
                                                                                    dones, system_infos):
                # append new samples to running paths
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["un_norm_rewards"].append(reward)
                running_paths[idx]["system_info"].append(system_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    if self.is_norm_reward:
                        rewards = utils.normalization(np.asarray(running_paths[idx]["un_norm_rewards"]))
                    else:
                        rewards = np.asarray(running_paths[idx]["un_norm_rewards"])
                    paths.append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        un_norm_rewards=np.asarray(running_paths[idx]["un_norm_rewards"]),
                        system_info = np.asarray(running_paths[idx]["system_info"]),
                        rewards=rewards
                    ))
                    new_samples += len(running_paths[idx]["un_norm_rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths


#process the sampled trajectory and calculate the advantanges
class MigrationSamplerProcess(SampleProcessor):
    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths: list of the trajectories
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert self.baseline, 'baseline must be specified'

        samples_data, paths = self._compute_samples_data(paths)

        # 7) compute normalized trajectory-batch rewards (for E-MAML)
        # overall_avg_reward = np.mean(samples_data['rewards'])
        # overall_avg_reward_std = np.std(samples_data['rewards'])
        #
        # samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (
        #         overall_avg_reward_std + 1e-8)

        self._log_path_stats(paths, log=log, log_prefix=log_prefix)

        return samples_data

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)


        # decide to normalize the return or not
        # 2) sample the values from the baseline
        # self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        paths = self._compute_advantages(paths, all_path_baselines)

        # calculate the return through bootstrapping instead of using Monte-Carlo method.
        for idx, path in enumerate(paths):
            path["returns"] = np.array(all_path_baselines[idx])

        observations, actions, shift_actions, rewards, returns, advantages, un_norm_rewards, system_info = self._append_path_data(paths)

        # normalize the returns

        # 5) if desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        returns = returns + advantages

        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            actions=actions,
            shift_actions = shift_actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            un_norm_rewards = un_norm_rewards,
            system_info = system_info
        )

        return samples_data, paths

    def _append_path_data(self, paths):
        observations = np.array([path["observations"] for path in paths], dtype=np.float32)
        actions = np.array([path["actions"] for path in paths], dtype=np.int32)
        rewards = np.array([path["rewards"] for path in paths], dtype=np.float32)
        returns = np.array([path["returns"] for path in paths], dtype=np.float32)
        advantages = np.array([path["advantages"] for path in paths], dtype=np.float32)
        un_norm_rewards = np.array([path["un_norm_rewards"] for path in paths], dtype=np.float32)
        system_info = np.array([path["system_info"] for path in paths], dtype=np.float32)

        shift_actions = np.array(np.column_stack((np.zeros(actions.shape[0],
                                            dtype=np.float32), actions[:, 0:-1])),
                                            dtype=np.float32)

        return observations, actions, shift_actions, rewards, returns, advantages, un_norm_rewards, system_info

class EvaluationSampler(Sampler):
    def __init__(self,
                 env,
                 policy,
                 batch_size,
                 max_path_length,
                 is_norm_reward = True):
        super(EvaluationSampler, self).__init__(env, policy, batch_size, max_path_length)
        self.total_timesteps_sampled = 0
        self.total_samples = batch_size * max_path_length
        self.is_batch = env.is_batch
        self.env = env
        self.is_norm_reward = is_norm_reward

    # evaluate the reward function
    def obtain_samples(self,
                       is_rnn=True,
                       is_greedy_sample=True):
        n_samples = 0
        if self.is_batch:
            reward_paths = [[] for _ in
                             range(self.env._num_traces)]
            system_info_paths = [[] for _ in
                             range(self.env._num_traces)]
            actions_paths = [[] for _ in
                             range(self.env._num_traces)]
        else:
            reward_paths = [[]]
            system_info_paths = [[]]
            actions_paths = [[]]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        # initial reset of envs
        obses = np.array(self.env.reset(), dtype=np.float32)
        if self.is_batch == False:
            obses = np.expand_dims(obses, axis=0)

        if is_rnn:
            hidden_state = self.policy.get_initial_hidden_state(obses)
            actions = np.zeros(shape=(obses.shape[0],), dtype=np.float32)

        reward_collects = []
        system_info_collects = []
        actions_info_collects = []

        while n_samples < self.total_samples:
            # step environments
            system_infos = self.env.current_system_state()
            t = time.time()


            if is_rnn:
                if is_greedy_sample:
                    actions, hidden_state = self.policy.greedy_sample(obses, actions, hidden_state)
                else:
                    actions, hidden_state = self.policy.sample(obses, actions, hidden_state)
            else:
                if is_greedy_sample:
                    actions = self.policy.greedy_sample(obses)
                else:
                    actions = self.policy.sample(obses)

            if len(np.array(actions).shape) == 0:
                actions = np.expand_dims(actions, axis=0)

            policy_time = time.time() - t

            next_obses, rewards, dones, env_infos = self.env.step(actions)
            if self.is_batch == False:
                next_obses = np.expand_dims(next_obses, axis=0)
                rewards = np.expand_dims(rewards, axis=0)
                system_infos = np.expand_dims(system_infos, axis=0)
                actions = np.expand_dims(actions, axis=0)
                dones = [dones]
                env_infos = [env_infos]

            env_time += time.time() - t

            new_samples = 0

            # when trajectory finish, initialize the hidden state
            if dones[0] and is_rnn:
                hidden_state = self.policy.get_initial_hidden_state(obses)

            for idx, observation, action, reward, env_info, done, system_info in zip(itertools.count(), obses, actions,
                                                                                     rewards, env_infos,
                                                                                     dones, system_infos):
                reward_paths[idx].append(reward)
                system_info_paths[idx].append(system_info)
                actions_paths[idx].append(action)

                if done:
                    reward_collects.append(np.asarray(reward_paths[idx]))
                    system_info_collects.append(np.asarray(system_info_paths[idx]))
                    actions_info_collects.append(np.asarray(actions_paths[idx]))

                    new_samples += len(reward_paths[idx])
                    reward_paths[idx] = []

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        return reward_collects, system_info_collects


class EvaluationSamplerForDRQN(Sampler):
    def __init__(self,
                 env,
                 policy,
                 batch_size,
                 max_path_length,
                 is_norm_reward = True,
                 is_rnn = True):
        super(EvaluationSamplerForDRQN, self).__init__(env, policy, batch_size, max_path_length)
        self.total_timesteps_sampled = 0
        self.total_samples = batch_size * max_path_length
        self.is_batch = env.is_batch
        self.env = env
        self.is_norm_reward = is_norm_reward
        self.is_rnn = is_rnn

    # evaluate the reward function
    def obtain_samples(self,
                       is_rnn=True):
        n_samples = 0

        reward_paths = [[] for _ in
                         range(self.env._num_traces)]
        system_info_paths = [[] for _ in
                         range(self.env._num_traces)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        # initial reset of envs
        obses = np.array(self.env.reset(), dtype=np.float32)
        if self.is_rnn:
            hidden_state = self.policy.get_initial_hidden_state(obses)

        reward_collects = []
        system_info_collects = []
        actions_paths_collects = []

        while n_samples < self.total_samples:
            # step environments

            t = time.time()

            if self.is_rnn:
                actions, hidden_state = self.policy.greedy_sample(obses, hidden_state)
            else:
                actions = self.policy.greedy_sample(obses)
            policy_time = time.time() - t

            next_obses, rewards, dones, env_infos = self.env.step(actions)
            system_infos = self.env.current_system_state()

            env_time += time.time() - t

            new_samples = 0

            # when trajectory finish, initialize the hidden state
            if dones[0] and self.is_rnn:
                hidden_state = self.policy.get_initial_hidden_state(obses)

            for idx, observation, action, reward, env_info, done, system_info in zip(itertools.count(), obses, actions,
                                                                                     rewards, env_infos,
                                                                                     dones, system_infos):
                reward_paths[idx].append(reward)
                system_info_paths[idx].append(system_info)


                if done:
                    reward_collects.append(np.asarray(reward_paths[idx]))
                    system_info_collects.append(np.asarray(system_info_paths[idx]))

                    new_samples += len(reward_paths[idx])
                    reward_paths[idx] = []

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        return reward_collects, system_info_collects


if __name__ == "__main__":
    from environment.migration_env import EnvironmentParameters
    from environment.migration_env import MigrationEnv
    from environment.batch_migration_env import EnvironmentParameters
    from environment.batch_migration_env import BatchMigrationEnv
    from baselines.linear_baseline import LinearFeatureBaseline
    from policies.fc_categorical_policy import FCCategoricalPolicy
    from policies.rnn_policy_with_action_input import RNNPolicyWithValue
    from baselines.rnn_critic_network_baseline import RNNCriticNetworkBaseline
    import tensorflow as tf

    possion_rate_vector = [18, 8, 17, 19, 10, 13, 19, 12, 8, 10, 14, 7, 17, 8, 11, 10, 16, 16, 9, 19, 20, 8, 15, 6,
                           6, 6, 17, 8, 17, 16, 15, 18, 8, 17, 5, 11, 12, 17, 10, 17, 12, 12, 9, 18, 7, 17, 9, 13,
                           8, 11, 12, 19, 11, 9, 5, 16, 9, 8, 10, 12, 20, 16, 8]

    # start point (41.856, 12.442), end point (41.928,12.5387), a region in Roman, Italy.
    env_default_parameters = EnvironmentParameters(trace_start_index=0,
                                                   num_traces=2,
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

    env = BatchMigrationEnv(env_default_parameters)

    policy = RNNPolicyWithValue(observation_dim=env._state_dim,
                                    action_dim=env._action_dim,
                                    rnn_parameter=256,
                                    embbeding_size=2)

    eval_sampler = EvaluationSampler(env,
                 policy=policy,
                   batch_size=10,
                   max_path_length=100)

    sampler = MigrationSampler(env,
                 policy=policy,
                   batch_size=240,
                   num_environment_per_core=4,
                   max_path_length=100,
                   parallel=True,
                   num_process=6)

    rewards, system_infos = eval_sampler.obtain_samples(is_rnn = True)

    print("shape of rewards: ", np.array(rewards).shape)
    print("shape of system_infos: ", np.array(system_infos).shape)

    print("evaluate_rewards:", np.mean(np.sum(rewards, axis=-1)))

    paths = sampler.obtain_samples(is_rnn=True)
    print("lenght of paths: ",len(paths))
    print("time step lengths: ", len(paths[0]['observations']))
    print("positions: ", [x[0] for x in paths[0]['observations']])
    print("system state: ", np.array(paths[0]['system_info']).shape)

    linear_feature_baseline = LinearFeatureBaseline()
    sampler_process = MigrationSamplerProcess(baseline=linear_feature_baseline,
                discount=0.99,
                gae_lambda=1,
                normalize_adv=False,
                positive_adv=False,)

    samples = sampler_process.process_samples(paths)
    print("processing sample's observation shape:", samples["observations"].shape)
    print("processing sample's advantage shape: ", samples["advantages"].shape)
    print("processing sample's system_info shape", samples["system_info"].shape)

    dataset = tf.data.Dataset.from_tensor_slices(samples).batch(25)

    for batch in dataset:
        print("old logits: ", batch["observations"].shape)
        print("iterator times: ")

    # test rnn sample
    # from policies.rnn_policy_with_action_input import RNNPolicy
    # rnn_policy = RNNPolicy(observation_dim=env._state_dim,
    #                        action_dim=env._action_dim,
    #                        rnn_parameter=128,
    #                        fc_parameters=128)
    # sampler = MigrationSampler(env,
    #                            policy=rnn_policy,
    #                            batch_size=240,
    #                            num_environment_per_core1=1,
    #                            max_path_length=100,
    #                            parallel=True,
    #                            num_process=6)
    # paths = sampler.obtain_samples(is_rnn=True)
    # print("lenght of paths: ", len(paths))
    # print("time step lengths: ", len(paths[0]['observations']))