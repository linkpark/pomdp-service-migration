from environment.batch_migration_env import BatchMigrationEnv
from environment.batch_migration_env import EnvironmentParameters
from sampler.migration_sampler import EvaluationSampler
from sampler.migration_sampler import MigrationSampler
from sampler.migration_sampler import MigrationSamplerProcess
from policies.always_migration_solution import always_migration_solution
from policies.optimal_solution import optimal_solution_for_batch_system_infos
from policies.no_migration_solution import no_migration_solution
import tensorflow as tf
import numpy as np


import utils.logger as logger
from sampler.migration_sampler import EvaluationSampler
from sampler.migration_sampler import EvaluationSamplerForDRQN
from sampler.migration_sampler import MigrationSampler
from sampler.migration_sampler import MigrationSamplerProcess
from baselines.rnn_critic_network_baseline import RNNCriticNetworkBaseline

from policies.random_solution import random_solution
from policies.optimal_solution import optimal_solution_for_batch_system_infos
from policies.no_migration_solution import no_migration_solution
from policies.fc_categorical_policy import FCCategoricalPolicyWithValue
from baselines.critic_network_baseline import CriticNetworkBaseline
from policies.always_migrate_policy import AlwaysMigratePolicy

from policies.q_network import QNetwork
from algorithms.dracm import DRACM
from dracm_trainer import Trainer
from algorithms.mab_ts import MABTSGuassianServiceMigration
from policies.rnn_q_network import RNNQNetwork
from policies.q_network import QNetwork
from policies.rnn_policy_with_action_input import RNNPolicyWithValue

logger.configure(dir="./log/test_migration_cost_rome", format_strs=['stdout', 'log', 'csv'])

number_of_base_state = 64
x_base_state = 8
y_base_state = 8

# possion_rate_vector = np.random.randint(15, 31, size=number_of_base_state)
# print("possion_rate_vector is: ", repr(possion_rate_vector))
possion_rate_vector = [7, 10, 8, 14, 15, 6, 20, 18, 11, 17, 20, 9, 8, 14, 9, 15, 8, 17, 9, 9, 10, 7, 17, 10,
                       13, 12, 5, 8, 10, 13, 19, 15, 10, 9, 10, 18, 12, 13, 5, 11, 7, 8, 8, 19, 15, 15, 6, 10,
                       5, 20, 17, 5, 5, 16, 5, 19, 19, 19, 9, 20, 17, 14, 17, 17]

# possion_rate_vector = [11, 8, 20, 9, 18, 18, 9, 17, 12, 17, 9, 17, 14, 10, 5, 7, 12,
#                        8, 20, 10, 14, 12, 20, 14, 8, 6, 15, 7, 18, 9, 8, 18, 17, 7,
#                        11, 11, 13, 14, 8, 18, 13, 17, 6, 18, 17, 18, 18, 7, 9, 6, 12,
#                        10, 9, 8, 20, 14, 11, 15, 14, 6, 6, 15, 16, 20]

# start point (41.856, 12.442), end point (41.928,12.5387), a region in Roman, Italy.
env_eval_parameters = EnvironmentParameters(trace_start_index=120,
                                            num_traces=30,
                                            server_frequency=128.0,  # GHz
                                            num_base_station=number_of_base_state,
                                            optical_fiber_trans_rate=500.0,
                                            backhaul_coefficient=0.02,
                                            migration_coefficient_low=1.0,
                                            migration_coefficient_high=3.0,
                                            server_poisson_rate=possion_rate_vector,
                                            client_poisson_rate=2,
                                            server_task_data_lower_bound=(0.05 * 1000.0 * 1000.0 * 8),
                                            server_task_data_higher_bound=(5 * 1000.0 * 1000.0 * 8),
                                            client_task_data_lower_bound=(0.05 * 1000.0 * 1000.0 * 8),
                                            client_task_data_higher_bound=(5 * 1000.0 * 1000.0 * 8),
                                            migration_size_low=0.5,
                                            migration_size_high=100.0,
                                            ratio_lower_bound=200.0,
                                            ratio_higher_bound=10000.0,
                                            map_width=8000.0, map_height=8000.0,
                                            num_horizon_servers=x_base_state, num_vertical_servers=y_base_state,
                                            traces_file_path='./environment/rome_traces_coordinate.txt',
                                            transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                            trace_length=100,
                                            trace_interval=12,
                                            is_full_observation=False,
                                            is_full_action=True)

env = BatchMigrationEnv(env_eval_parameters)
eval_sample_size = 30

rnn_policy = RNNPolicyWithValue(observation_dim=env._state_dim,
                                action_dim=env._action_dim,
                                rnn_parameter=256,
                                embbeding_size=2)
vf_baseline = RNNCriticNetworkBaseline(rnn_policy)

eval_sampler = EvaluationSampler(env,
                                 policy=rnn_policy,
                                 batch_size=eval_sample_size,
                                 max_path_length=100)

sampler_process = MigrationSamplerProcess(baseline=vf_baseline,
                                          discount=0.99,
                                          gae_lambda=0.95,
                                          normalize_adv=True,
                                          positive_adv=False)

fc_policy = FCCategoricalPolicyWithValue(observation_dim=env._state_dim,
                                         action_dim=env._action_dim,
                                         fc_parameters=[256])
vf_baseline = CriticNetworkBaseline(fc_policy)


fc_eval_sampler = EvaluationSampler(env,
                                    policy=fc_policy,
                                    batch_size=10,
                                    max_path_length=100)

algo = DRACM(policy=rnn_policy,
             value_function=rnn_policy,
             policy_optimizer=tf.keras.optimizers.Adam(5e-4),
             value_optimizer=tf.keras.optimizers.Adam(5e-4),
             is_rnn=True,
             is_shared_critic_net=True,
             num_inner_grad_steps=4,
             clip_value=0.2,
             vf_coef=0.5,
             max_grad_norm=0.5,
             entropy_coef=0.01)


rnn_q_network = RNNQNetwork(observation_dim=env._state_dim,
                            action_dim=env._action_dim,
                            rnn_parameter=256,
                            fc_parameters=128,
                            epsilon=0.1)

rnn_q_net_sampler = EvaluationSamplerForDRQN(env,
                                             policy=rnn_q_network,
                                             batch_size=eval_sample_size,
                                             max_path_length=100)

q_network = QNetwork(observation_dim=env._state_dim,
                     action_dim=env._action_dim,
                     hidden_parameter=256,
                     fc_parameters=128,
                     epsilon=0.1)

q_network_eval_sampler = EvaluationSamplerForDRQN(env,
                                                  policy=q_network,
                                                  batch_size=30,
                                                  max_path_length=100,
                                                  is_rnn=False)

dracm_model_path = "./checkpoints_rome/checkpoints_ppo_64-bs-new-2/model_checkpoint_epoch_115"
fc_dracm_model_path = "./checkpoints_rome/checkpoints_ppo_64-bs-new-no-rnn/model_checkpoint_epoch_115"
drqn_model_path = "./checkpoints_rome/checkpoints_drqn_rome_64-bs-new/model_checkpoint_3800"
dqn_model_path = "./checkpoints_rome/checkpoints_dqn_rome_64-bs-new/model_checkpoint_3800"

rnn_policy.load_weights(dracm_model_path)
logger.log("Load rnn model successfully....")


fc_policy.load_weights(fc_dracm_model_path)
logger.log("Load fc model successfully....")


rnn_q_network.load_weights(drqn_model_path)
logger.log("Load rnn q network model successfully ....")

q_network.load_weights(dqn_model_path)
logger.log("Load q network model successfully ....")

migration_co_set = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
for migration_co in migration_co_set:
    print("migration coefficient: ", migration_co)
    env.migration_coefficient_high = migration_co
    env.migration_coefficient_low = migration_co
    am_eval_sampler = EvaluationSampler(env,
                      policy=AlwaysMigratePolicy(env._state_dim,action_dim=env._action_dim),
                      batch_size=30,
                      max_path_length=100)

    rewards, system_infos = am_eval_sampler.obtain_samples(is_rnn=False)
    system_infos = np.array(system_infos)
    logger.log("processing sample's system_info shape", system_infos.shape)


    always_migration_latency = always_migration_solution(env, system_infos)
    logger.log("always migration latency is: ", always_migration_latency)

    never_migration_latency = no_migration_solution(env, system_infos)
    logger.log("no migration latency is: ", never_migration_latency)

    optimal_rewards = optimal_solution_for_batch_system_infos(env, system_infos)
    logger.log("optimal latency is: ", optimal_rewards)

    random_rewards = random_solution(env, system_infos)
    logger.log("random latency is: ", random_rewards)

    dqn_rewards_collects, _ = q_network_eval_sampler.obtain_samples(is_rnn=False)
    drqn_rewards_collects, _ = rnn_q_net_sampler.obtain_samples(is_rnn=True)
    fc_reward_collects, _ = fc_eval_sampler.obtain_samples(is_rnn=False, is_greedy_sample=False)
    sample_reward_collects, _ = eval_sampler.obtain_samples(is_rnn=rnn_policy,
                                                                             is_greedy_sample=False)
    reward_collects, system_info_collects = eval_sampler.obtain_samples(is_rnn=rnn_policy,
                                                                             is_greedy_sample=True)

    env.reset()
    mab_ts_algo = MABTSGuassianServiceMigration(env)
    totoal_rewards = mab_ts_algo.train(num_iteration=3)
    mab_ts_rewards = totoal_rewards[-1]

    dqn_rewards = np.mean(np.sum(dqn_rewards_collects, axis=-1))
    drqn_rewards = np.mean(np.sum(drqn_rewards_collects, axis=-1))
    fc_ppo_rewards = np.mean(np.sum(fc_reward_collects, axis=-1))
    ppo_rewards = np.mean(np.sum(reward_collects, axis=-1))
    ppo_sample_rewards = np.mean(np.sum(sample_reward_collects, axis=-1))

    logger.log("eval fc ppo latency ", -fc_ppo_rewards)
    logger.log("eval dqn latency ", -dqn_rewards)
    logger.log("eval drqn latency ", -drqn_rewards)
    logger.log("eval sample latency ", -ppo_sample_rewards)
    logger.log("eval latency ", -ppo_rewards)
    logger.log("eval mab-ts reward ", -mab_ts_rewards)