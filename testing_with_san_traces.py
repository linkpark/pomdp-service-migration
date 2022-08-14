import tensorflow as tf
import numpy as np

import utils.logger as logger
from sampler.migration_sampler import EvaluationSampler
from sampler.migration_sampler import EvaluationSamplerForDRQN
from sampler.migration_sampler import MigrationSampler
from sampler.migration_sampler import MigrationSamplerProcess
from baselines.rnn_critic_network_baseline import RNNCriticNetworkBaseline

from policies.random_solution import random_solution
from policies.always_migration_solution import always_migration_solution
from policies.optimal_solution import optimal_solution_for_batch_system_infos
from policies.no_migration_solution import no_migration_solution
from policies.fc_categorical_policy import FCCategoricalPolicyWithValue
from baselines.critic_network_baseline import CriticNetworkBaseline

from policies.q_network import QNetwork
from algorithms.dracm import DRACM
from dracm_trainer import Trainer
from algorithms.mab_ts import MABTSGuassianServiceMigration

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class Tester(object):
    def __init__(self,
                 testing_env,
                 policy,
                 fc_policy,
                 rnn_q_net_policy,
                 q_net_policy,
                 eval_sampler,
                 fc_eval_sampler,
                 rnn_q_net_eval_sampler,
                 q_net_eval_sampler):
        self.eval_env = testing_env
        self.policy = policy
        self.fc_policy = fc_policy
        self.rnn_q_net_policy = rnn_q_net_policy
        self.q_net_policy = q_net_policy
        self.eval_sampler = eval_sampler
        self.fc_eval_sampler = fc_eval_sampler
        self.rnn_q_net_eval_sampler = rnn_q_net_eval_sampler
        self.q_net_eval_sampler = q_net_eval_sampler
        self.mab_ts_algo = MABTSGuassianServiceMigration(testing_env)

    def run_test(self, rnn_policy=True):


        avg_ppo_rewards = 0.0
        avg_fc_ppo_rewards = 0.0
        avg_random_rewards = 0.0
        avg_always_migrate_rewards = 0.0
        avg_optimal_rewards = 0.0
        avg_no_migration_rewards = 0.0
        avg_ppo_sample_rewards = 0.0
        avg_drqn_rewards = 0.0
        avg_dqn_rewards = 0.0
        avg_mab_ts_rewards = 0.0

        iter_number = 1
        for i in range(iter_number):
            dqn_rewards_collects, _ = self.q_net_eval_sampler.obtain_samples(is_rnn=False)
            drqn_rewards_collects, _ = self.rnn_q_net_eval_sampler.obtain_samples(is_rnn=rnn_policy)
            sample_reward_collects, _ = self.eval_sampler.obtain_samples(is_rnn=rnn_policy, is_greedy_sample=False)
            fc_reward_collects, _ = self.fc_eval_sampler.obtain_samples(is_rnn=False, is_greedy_sample=False)
            reward_collects, system_info_collects = self.eval_sampler.obtain_samples(is_rnn=rnn_policy,
                                                                                     is_greedy_sample=True)

            dqn_rewards = np.mean(np.sum(dqn_rewards_collects, axis=-1))
            drqn_rewards = np.mean(np.sum(drqn_rewards_collects, axis=-1))
            fc_ppo_rewards = np.mean(np.sum(fc_reward_collects, axis=-1))
            ppo_rewards = np.mean(np.sum(reward_collects, axis=-1))
            ppo_sample_rewards = np.mean(np.sum(sample_reward_collects, axis=-1))

            random_rewards = random_solution(self.eval_sampler.env, system_info_collects)
            always_migrate_rewards = always_migration_solution(self.eval_sampler.env, system_info_collects)
            optimal_rewards = optimal_solution_for_batch_system_infos(self.eval_sampler.env, system_info_collects)
            # optimal_rewards = 0.0
            no_migration_rewards = no_migration_solution(self.eval_sampler.env, system_info_collects)

            self.eval_env.reset()
            totoal_rewards = self.mab_ts_algo.train(num_iteration=3)
            # totoal_rewards = np.array([0.0])
            mab_ts_rewards = totoal_rewards[-1]

            logger.log("---- round " + str(i) + " ----")
            logger.log("eval sample reward", ppo_sample_rewards)
            logger.log("eval fc ppo reward", fc_ppo_rewards)
            logger.log("eval dqn reward", dqn_rewards)
            logger.log("eval drqn reward", drqn_rewards)
            logger.log("eval reward", ppo_rewards)
            logger.log("eval random reward", -random_rewards)
            logger.log("eval always migration reward", -always_migrate_rewards)
            logger.log("eval optimal reward", -optimal_rewards)
            logger.log("eval mab-ts reward", -mab_ts_rewards)
            logger.log("no_migration_solution", -no_migration_rewards)

            avg_dqn_rewards += dqn_rewards
            avg_ppo_rewards += ppo_rewards
            avg_fc_ppo_rewards += fc_ppo_rewards
            avg_drqn_rewards += drqn_rewards
            avg_random_rewards += random_rewards
            avg_always_migrate_rewards += always_migrate_rewards
            avg_optimal_rewards += optimal_rewards
            avg_no_migration_rewards += no_migration_rewards
            avg_ppo_sample_rewards += ppo_sample_rewards
            avg_mab_ts_rewards += mab_ts_rewards

        logger.logkv("eval sample reward", avg_ppo_sample_rewards / iter_number)
        logger.logkv("eval reward", avg_ppo_rewards / iter_number)
        logger.logkv("eval fc ppo reward", avg_fc_ppo_rewards / iter_number)
        logger.logkv("eval dqn reward", avg_dqn_rewards / iter_number)
        logger.logkv("eval drqn reward", avg_drqn_rewards / iter_number)
        logger.logkv("eval random reward", -(avg_random_rewards / iter_number))
        logger.logkv("eval always migration reward", -(avg_always_migrate_rewards / iter_number))
        logger.logkv("eval optimal reward", -(avg_optimal_rewards / iter_number))
        logger.logkv("eval no migration reward", -(avg_no_migration_rewards / iter_number))
        logger.logkv("eval mab-ts reward", -(avg_mab_ts_rewards / iter_number))

        logger.dumpkvs()


if __name__ == "__main__":
    from environment.batch_migration_env import EnvironmentParameters
    from environment.batch_migration_env import BatchMigrationEnv
    from policies.rnn_q_network import RNNQNetwork
    from policies.q_network import QNetwork
    from policies.rnn_policy_with_action_input import RNNPolicyWithValue
    from algorithms.mab_ts import MABTSGuassianServiceMigration

    logger.configure(dir="./log/test_with_trace_number_30", format_strs=['stdout', 'log', 'csv'])

    number_of_base_state = 64
    x_base_state = 8
    y_base_state = 8

    # possion_rate_vector = np.random.randint(15, 31, size=number_of_base_state)
    # print("possion_rate_vector is: ", repr(possion_rate_vector))
    possion_rate_vector = [11, 8, 20, 9, 18, 18, 9, 17, 12, 17, 9, 17, 14, 10, 5, 7, 12,
                           8, 20, 10, 14, 12, 20, 14, 8, 6, 15, 7, 18, 9, 8, 18, 17, 7,
                           11, 11, 13, 14, 8, 18, 13, 17, 6, 18, 17, 18, 18, 7, 9, 6, 12,
                           10, 9, 8, 20, 14, 11, 15, 14, 6, 6, 15, 16, 20]

    # start point (41.856, 12.442), end point (41.928,12.5387), a region in Roman, Italy.
    env_eval_parameters =EnvironmentParameters(trace_start_index=120,
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
    env = BatchMigrationEnv(env_eval_parameters)
    eval_sample_size = 30

    print("action dim of the environment: ", env._action_dim)

    rnn_policy = RNNPolicyWithValue(observation_dim=env._state_dim,
                                    action_dim=env._action_dim,
                                    rnn_parameter=256,
                                    embbeding_size=2)
    vf_baseline = RNNCriticNetworkBaseline(rnn_policy)

    sampler = MigrationSampler(env,
                               policy=rnn_policy,
                               batch_size=400,
                               num_environment_per_core=2,
                               max_path_length=100,
                               parallel=True,
                               num_process=5,
                               is_norm_reward=False)

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

    fc_sampler = MigrationSampler(env,
                               policy=fc_policy,
                               batch_size=4800,
                               num_environment_per_core=2,
                               max_path_length=100,
                               parallel=True,
                               num_process=8,
                               is_norm_reward=True)  # 2 * 4 * 30

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

    trainer = Trainer(train_env=env,
                      eval_env=env,
                      algo=algo,
                      sampler=sampler,
                      sample_processor=sampler_process,
                      update_batch_size=100,
                      policy=rnn_policy,
                      n_itr=10,
                      save_interval=5,
                      eval_sampler=eval_sampler,
                      test_interval=10,
                      save_path=None)

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

    tester = Tester(testing_env=env,
                    policy=rnn_policy,
                    fc_policy=fc_policy,
                    rnn_q_net_policy=rnn_q_network,
                    q_net_policy=q_network,
                    eval_sampler=eval_sampler,
                    fc_eval_sampler=fc_eval_sampler,
                    rnn_q_net_eval_sampler=rnn_q_net_sampler,
                    q_net_eval_sampler=q_network_eval_sampler)


    dracm_model_path = "./checkpoints_san/checkpoints_ppo_64-bs-new-2/model_checkpoint_epoch_115"
    fc_dracm_model_path = "./checkpoints_san/checkpoints_ppo_64-bs-new-2-no-rnn/model_checkpoint_epoch_115"
    drqn_model_path = "./checkpoints_san/checkpoints_drqn_san_64-bs-new/model_checkpoint_3800"
    dqn_model_path = "./checkpoints_san/checkpoints_dqn_san_64-bs-new/model_checkpoint_3800"

    if dracm_model_path != None:
        rnn_policy.load_weights(dracm_model_path)
        logger.log("Load rnn model successfully....")

    if fc_dracm_model_path != None:
        fc_policy.load_weights(fc_dracm_model_path)
        logger.log("Load fc model successfully....")

    if drqn_model_path != None:
        rnn_q_network.load_weights(drqn_model_path)
        logger.log("Load rnn q network model successfully ....")

    if dqn_model_path != None:
        q_network.load_weights(dqn_model_path)
        logger.log("Load q network model successfully ....")

    migration_set = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    for migration_coeff in migration_set:
        env.migration_coefficient_high = migration_coeff
        env.migration_coefficient_low = migration_coeff

        tester.run_test(rnn_policy=True)
