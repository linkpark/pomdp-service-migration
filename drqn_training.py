from environment.migration_env import EnvironmentParameters
from environment.migration_env import MigrationEnv
from environment.batch_migration_env import BatchMigrationEnv
from policies.rnn_q_network import RNNQNetwork
from sampler.replay_buffer import SequentialReplayBuffer
from sampler.migration_sampler import MigrationSamplerForDRQN
from sampler.migration_sampler import EvaluationSamplerForDRQN
from algorithms.drqn import DRQN

import utils.logger as logger
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# training the drqn with rome traces
logger.configure(dir="./log/drqn_training", format_strs=['stdout', 'log', 'csv'])

number_of_base_state = 64
x_base_state = 8
y_base_state = 8

possion_rate_vector = [14, 12, 15, 14, 24, 23, 13, 10, 20, 26, 18, 25, 27, 19, 13, 27, 28,
            10, 12, 24, 26, 29, 23, 27, 22, 12, 22, 17, 13, 29, 21, 30, 13, 28,
            22, 22, 27, 24, 28, 21, 29, 23, 15, 12, 26, 30, 19, 10, 28, 28, 27,
            19, 30, 26, 11, 27, 18, 10, 19, 30, 17, 23, 21, 13]

# start point (41.856, 12.442), end point (41.928,12.5387), a region in Roman, Italy.
env_default_parameters = EnvironmentParameters(trace_start_index=0,
                                                   num_traces=100,
                                                   server_frequency=480.0,  # GHz
                                                   num_base_station=number_of_base_state,
                                                   optical_fiber_trans_rate=1000.0,
                                                   server_poisson_rate=possion_rate_vector, client_poisson_rate=4,
                                                   server_task_data_lower_bound=(3 * 1024.0 * 1024.0 * 8),
                                                   server_task_data_higher_bound=(10 * 1024.0 * 1024.0 * 8),
                                                   client_task_data_lower_bound=(3 * 1024.0 * 1024.0 * 8),
                                                   client_task_data_higher_bound=(10 * 1024.0 * 1024.0 * 8),
                                                   migration_cost_low=0.5,
                                                   migration_cost_high=3.0,
                                                   ratio_lower_bound=200.0,
                                                   ratio_higher_bound=10000.0,
                                                   map_width=8000.0, map_height=8000.0,
                                                   num_horizon_servers=x_base_state, num_vertical_servers=y_base_state,
                                                   traces_file_path='./environment/rome_traces_coordinate.txt',
                                                   transmission_rates=[100.0, 80.0, 60.0, 40.0, 20.0],  # Mbps
                                                   trace_length=100,
                                                   trace_interval=12, # time_slots interval 180s = 3min
                                                   is_full_observation=False,
                                                   is_full_action=True)

env_eval_parameters = EnvironmentParameters(trace_start_index=120,
                                                   num_traces=30,
                                                   server_frequency=480.0,  # GHz
                                                   num_base_station=number_of_base_state,
                                                   optical_fiber_trans_rate=1000.0,
                                                   server_poisson_rate=possion_rate_vector, client_poisson_rate=4,
                                                   server_task_data_lower_bound=(3 * 1024.0 * 1024.0 * 8),
                                                   server_task_data_higher_bound=(10 * 1024.0 * 1024.0 * 8),
                                                   client_task_data_lower_bound=(3 * 1024.0 * 1024.0 * 8),
                                                   client_task_data_higher_bound=(10 * 1024.0 * 1024.0 * 8),
                                                   migration_cost_low=0.5,
                                                   migration_cost_high=3.0,
                                                   ratio_lower_bound=200.0,
                                                   ratio_higher_bound=10000.0,
                                                   map_width=8000.0, map_height=8000.0,
                                                   num_horizon_servers=x_base_state, num_vertical_servers=y_base_state,
                                                   traces_file_path='./environment/rome_traces_coordinate.txt',
                                                   transmission_rates=[100.0, 80.0, 60.0, 40.0, 20.0],  # Mbps
                                                   trace_length=100,
                                                   trace_interval=12,
                                                   is_full_observation=False,
                                                   is_full_action=True)

env = BatchMigrationEnv(env_default_parameters)
eval_env = BatchMigrationEnv(env_eval_parameters)

logger.log("initializing environment complete")

q_network = RNNQNetwork(observation_dim=env._state_dim,
                            action_dim=env._action_dim,
                            rnn_parameter=256,
                            fc_parameters=128,
                            epsilon=0.1)


sampler = MigrationSamplerForDRQN(env,
                           policy=q_network,
                              batch_size=4800,
                              num_environment_per_core=2,
                              max_path_length=100,
                              parallel=True,
                              num_process=8,
                              is_norm_reward=True)

eval_sampler = EvaluationSamplerForDRQN(eval_env,
                                 policy=q_network,
                                 batch_size=40,
                                 max_path_length=100)

replay_buffer = SequentialReplayBuffer(size=9600)

paths = sampler.obtain_samples(epsilon=1.0)
replay_buffer.add(paths)

print("replay buffer size is: ", replay_buffer.size())

drqn_agent = DRQN(
                 q_network,
                 replay_buffer,
                 sampler,
                 action_dim=env._action_dim,
                 gamma=0.99,
                 optimizer=tf.keras.optimizers.Adam(1e-3),
                 log_interval=40,
                 eval_sampler=eval_sampler,
                 save_interval=200,
                 sample_interval=100)

drqn_agent.train(4000, 480)