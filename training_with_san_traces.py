from environment.batch_migration_env import EnvironmentParameters
from environment.migration_env import MigrationEnv
from environment.batch_migration_env import BatchMigrationEnv
from baselines.linear_baseline import LinearTimeBaseline
from baselines.rnn_critic_network_baseline import RNNCriticNetworkBaseline
from policies.rnn_policy_with_action_input import RNNPolicyWithValue


from sampler.migration_sampler import MigrationSamplerProcess
from sampler.migration_sampler import MigrationSampler
from sampler.migration_sampler import EvaluationSampler
from algorithms.dracm import DRACM
from dracm_trainer import Trainer

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
    logger.configure(dir="./log/ppo-san-with-optimal-100-bs-64-new-50", format_strs=['stdout', 'log', 'csv'])

    # bs number = 64
    possion_rate_vector = [11,  8, 20,  9, 18, 18,  9, 17, 12, 17,  9, 17, 14, 10,  5,  7, 12,
        8, 20, 10, 14, 12, 20, 14,  8,  6, 15,  7, 18,  9,  8, 18, 17,  7,
       11, 11, 13, 14,  8, 18, 13, 17,  6, 18, 17, 18, 18,  7,  9,  6, 12,
       10,  9,  8, 20, 14, 11, 15, 14,  6,  6, 15, 16, 20]

    env_default_parameters = EnvironmentParameters(trace_start_index=0,
                                                num_traces=100,
                                                server_frequency=128.0,  # GHz
                                                num_base_station=number_of_base_state,
                                                optical_fiber_trans_rate=500.0,
                                                backhaul_coefficient=0.02,
                                                migration_coefficient_low=1.0,
                                                migration_coefficient_high =3.0,
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

    print("action dim of the environment: ", env._action_dim)

    rnn_policy = RNNPolicyWithValue(observation_dim=env._state_dim,
                                    action_dim=env._action_dim,
                                    rnn_parameter=256,
                                    embbeding_size=2)
    vf_baseline = RNNCriticNetworkBaseline(rnn_policy)

    sampler = MigrationSampler(env,
                               policy=rnn_policy,
                               batch_size=4800,
                               num_environment_per_core=1,
                               max_path_length=100,
                               parallel=True,
                               num_process=8,
                               is_norm_reward=True)

    eval_sampler = EvaluationSampler(eval_env,
                                     policy=rnn_policy,
                                     batch_size=10,
                                     max_path_length=100)

    sampler_process = MigrationSamplerProcess(baseline=vf_baseline,
                                              discount=0.99,
                                              gae_lambda=0.95,
                                              normalize_adv=True,
                                              positive_adv=False)
    algo = DRACM(policy=rnn_policy,
                 value_function=rnn_policy,
                 policy_optimizer=tf.keras.optimizers.Adam(1e-3),
                 value_optimizer=tf.keras.optimizers.Adam(1e-3),
                 is_rnn=True,
                 is_shared_critic_net=True,
                 num_inner_grad_steps=4,
                 clip_value=0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 entropy_coef=0.01)

    trainer = Trainer(train_env=env,
                      eval_env=eval_env,
                      algo=algo,
                      sampler=sampler,
                      sample_processor=sampler_process,
                      update_batch_size=480,
                      policy=rnn_policy,
                      n_itr=120,
                      save_interval=5,
                      eval_sampler=eval_sampler,
                      test_interval=10,
                      save_path='./checkpoints_san_64-bs-new-50/model_checkpoint_epoch_')

    trainer.train(rnn_policy=True, is_test=False)