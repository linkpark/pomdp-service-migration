import numpy as np
import time
from utils import logger
from policies.random_migrate_policy import RandomMigratePolicy
from policies.always_migrate_policy import AlwaysMigratePolicy

from dracm_trainer import Trainer
from environment.migration_env import EnvironmentParameters
from environment.migration_env import MigrationEnv
from environment.batch_migration_env import BatchMigrationEnv
from baselines.critic_network_baseline import CriticNetworkBaseline
from baselines.linear_baseline import LinearTimeBaseline
from baselines.rnn_critic_network_baseline import RNNCriticNetworkBaseline
from policies.rnn_policy_with_action_input import RNNPolicy
from policies.rnn_policy_with_action_input import RNNValueNet
from policies.optimal_solution import optimal_solution_for_batch_system_infos

from policies.fc_categorical_policy import FCCategoricalPolicy
from baselines.linear_baseline import LinearFeatureBaseline
from policies.fc_categorical_policy import FCCategoricalPolicyWithValue
from policies.fc_categorical_policy import FCValueNetwork
from policies.rnn_critic_network import RNNValueNetwork

from sampler.migration_sampler import MigrationSamplerProcess
from sampler.migration_sampler import MigrationSampler
from algorithms.dracm import DRACM
import tensorflow as tf

logger.configure(dir="./log/pomdp-with-fc-linear-baseline-ppo-results-with-optimal", format_strs=['stdout', 'log', 'csv'])

server_poisson_rate = [18,  8, 17, 19, 10, 13, 19, 12 , 8 ,10, 14 , 7, 17,  8, 11, 10, 16, 16,  9, 19 ,20,  8, 15,  6,
  6,  6, 17,  8, 17, 16, 15, 18,  8, 17,  5, 11, 12, 17, 10, 17, 12, 12,  9, 18,  7, 17,  9, 13,
  8, 11, 12, 19, 11,  9,  5, 16,  9,  8, 10, 12, 20, 16,  8]

env_default_parameters = EnvironmentParameters(num_traces=10,
                                               num_base_station=63, optical_fiber_trans_rate=60.0,
                                               server_poisson_rate=server_poisson_rate, client_poisson_rate=4,
                                               server_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                               server_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                               ratio_lower_bound=100.0,
                                               client_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                               client_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                               ratio_higher_bound=3200.0, map_width=4500.0, map_height=3500.0,
                                               num_horizon_servers=9, num_vertical_servers=7,
                                               traces_file_path='./environment/default_scenario_LocationSnapshotReport.txt',
                                               transmission_rates=[20.0, 16.0, 12.0, 8.0, 4.0],
                                               trace_length=100,
                                               is_full_observation=False,
                                               is_full_action=True)

#env = BatchMigrationEnv(env_default_parameters)
env = MigrationEnv(env_default_parameters)

fc_policy = FCCategoricalPolicy(observation_dim=env._state_dim,
                                action_dim=env._action_dim,
                                fc_parameters=[128, 64, 32])

fc_critic = FCValueNetwork(observation_dim=env._state_dim,
                                fc_parameters=[128, 64, 32])


baseline = CriticNetworkBaseline(critic_network=fc_critic)
#baseline = LinearFeatureBaseline()

sampler = MigrationSampler(env,
                           policy=fc_policy,
                           batch_size=160,
                           num_environment_per_core=4,
                           max_path_length=100,
                           parallel=True,
                           num_process=10,
                           is_norm_reward=False)


sampler_process = MigrationSamplerProcess(baseline=baseline,
                                          discount=0.99,
                                          gae_lambda=0.95,
                                          normalize_adv=True,
                                          positive_adv=False)

algo = DRACM(policy = fc_policy,
             value_function = fc_critic,
             policy_optimizer = tf.keras.optimizers.Adam(1e-3),
             value_optimizer= tf.keras.optimizers.Adam(1e-3),
             is_rnn=False,
             num_inner_grad_steps=24,
             clip_value=0.2,
             vf_coef=0.5,
             max_grad_norm=0.5,
             entropy_coef = 0.01)

trainer = Trainer(train_env = env,
             algo=algo,
             sampler = sampler,
             sample_processor = sampler_process,
             update_batch_size = 32,
             policy = fc_policy,
             n_itr = 400,
             save_interval = 1)

eval_sampler_1 = MigrationSampler(env,
                                  policy=RandomMigratePolicy(observation_dim=env._state_dim,
                                                             action_dim=env._action_dim),
                                  batch_size=160,
                                  num_environment_per_core=4,
                                  max_path_length=100,
                                  parallel=True,
                                  num_process=5)

eval_sampler_2 = MigrationSampler(env,
                                  policy=AlwaysMigratePolicy(env._state_dim,
                                                             action_dim=env._action_dim),
                                  batch_size=160,
                                  num_environment_per_core=4,
                                  max_path_length=100,
                                  parallel=True,
                                  num_process=5)

baseline = LinearTimeBaseline()
eval_sample_processor = MigrationSamplerProcess(baseline=baseline,
                                          discount=0.99,
                                          gae_lambda=0.95,
                                          normalize_adv=True,
                                          positive_adv=False )

avg_random_rewards = 0.0
avg_always_migrate_rewards = 0.0
logger.log("evaluate random policy ....")
eval_paths_1 = eval_sampler_1.obtain_samples(log=False, log_prefix='')
eval_samples_1 = eval_sample_processor.process_samples(eval_paths_1)
eval_ret_1 = np.sum(eval_samples_1["un_norm_rewards"], axis=-1)
avg_random_rewards = np.mean(eval_ret_1)

logger.log("evaluate always migrate policy ....")
eval_paths_2 = eval_sampler_2.obtain_samples(log=False, log_prefix='')
eval_samples_2 = eval_sample_processor.process_samples(eval_paths_2)
eval_ret_2 = np.sum(eval_samples_2["un_norm_rewards"], axis=-1)
avg_always_migrate_rewards = np.mean(eval_ret_2)

optimal_migrate_rewards = optimal_solution_for_batch_system_infos(env, eval_samples_2["system_info"])

trainer.train(rnn_policy=False,
              avg_random_rewards=avg_random_rewards,
              avg_always_migrate_rewards=avg_always_migrate_rewards,
              optimal_migrate_rewards = optimal_migrate_rewards)