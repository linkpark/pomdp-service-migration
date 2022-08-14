from environment.batch_migration_env import EnvironmentParameters
from environment.migration_env import MigrationEnv
from environment.batch_migration_env import BatchMigrationEnv

from policies.q_network import QNetwork
from sampler.replay_buffer import SequentialReplayBuffer
from sampler.migration_sampler import MigrationSamplerForDRQN
from sampler.migration_sampler import EvaluationSamplerForDRQN
from algorithms.dqn import DQN

import tensorflow as tf
import numpy as np

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from utils import logger

if __name__ == "__main__":
    number_of_base_state = 64
    x_base_state = 8
    y_base_state = 8

    # original point = (37.70957, -122.48302)

    # possion_rate_vector = np.random.randint(10, 31, size=number_of_base_state)
    # print("possion_rate_vector is: ", repr(possion_rate_vector))

    # 40.0, 36.0, 32.0, 28.0, 24.0,
    logger.configure(dir="./log/dqn-san-with-100-bs-64-new", format_strs=['stdout', 'log', 'csv'])

    # bs number = 64
    possion_rate_vector = [11, 8, 20, 9, 18, 18, 9, 17, 12, 17, 9, 17, 14, 10, 5, 7, 12,
                           8, 20, 10, 14, 12, 20, 14, 8, 6, 15, 7, 18, 9, 8, 18, 17, 7,
                           11, 11, 13, 14, 8, 18, 13, 17, 6, 18, 17, 18, 18, 7, 9, 6, 12,
                           10, 9, 8, 20, 14, 11, 15, 14, 6, 6, 15, 16, 20]

    env_default_parameters = EnvironmentParameters(trace_start_index=0,
                                                   num_traces=100,
                                                   server_frequency=128.0,  # GHz
                                                   num_base_station=number_of_base_state,
                                                   optical_fiber_trans_rate=500.0,
                                                   backhaul_coefficient=0.02,
                                                   migration_coefficient_low=1.0,
                                                   migration_coefficient_high=3.0,
                                                   server_poisson_rate=possion_rate_vector, client_poisson_rate=2,
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
                                                   traces_file_path='./environment/san_traces_coordinate.txt',
                                                   transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                                   trace_length=100,
                                                   trace_interval=3,
                                                   is_full_observation=False,
                                                   is_full_action=True)

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
                                                traces_file_path='./environment/san_traces_coordinate.txt',
                                                transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                                trace_length=100,
                                                trace_interval=3,
                                                is_full_observation=False,
                                                is_full_action=True)

    env = BatchMigrationEnv(env_default_parameters)
    eval_env = BatchMigrationEnv(env_eval_parameters)

    q_network = QNetwork(observation_dim=env._state_dim,
                            action_dim=env._action_dim,
                            hidden_parameter=256,
                            fc_parameters=128,
                            epsilon=0.1)

    sampler = MigrationSamplerForDRQN(env,
                                      policy=q_network,
                                      batch_size=4800,
                                      num_environment_per_core=2,
                                      max_path_length=100,
                                      parallel=True,
                                      num_process=8,
                                      is_norm_reward=True,
                                      is_rnn=False)

    eval_sampler = EvaluationSamplerForDRQN(eval_env,
                                            policy=q_network,
                                            batch_size=30,
                                            max_path_length=100,
                                            is_rnn=False)

    replay_buffer = SequentialReplayBuffer(size=9600)


    # start from epsion = 1.0 and decay with training
    paths = sampler.obtain_samples(epsilon=1.0)
    replay_buffer.add(paths)

    print("replay buffer size is: ", replay_buffer.size())

    drqn_agent = DQN(
        q_network,
        replay_buffer,
        sampler,
        action_dim=env._action_dim,
        gamma=0.99,
        optimizer=tf.keras.optimizers.Adam(5e-4),
        log_interval=40,
        eval_sampler=eval_sampler,
        save_interval=200,
        sample_interval=100,
        model_path="checkpoints_dqn_san_64-bs-new/model_checkpoint")

    drqn_agent.train(4000, 480)