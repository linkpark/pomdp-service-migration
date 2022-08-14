import tensorflow as tf
import numpy as np

import utils.logger as logger
from sampler.migration_sampler import EvaluationSampler
from sampler.migration_sampler import EvaluationSamplerForDRQN

from policies.random_solution import random_solution
from policies.always_migration_solution import always_migration_solution
from policies.optimal_solution import optimal_solution_for_batch_system_infos
from policies.no_migration_solution import no_migration_solution

class Tester(object):
    def __init__(self,
                 testing_env,
                 policy,
                 q_net_policy,
                 eval_sampler,
                 q_net_eval_sampler):
        self.eval_env = testing_env
        self.policy = policy
        self.q_net_policy = q_net_policy
        self.eval_sampler = eval_sampler
        self.q_net_eval_sampler = q_net_eval_sampler

    def run_test(self, rnn_model_path=None, q_net_model_path=None, rnn_policy=True):
        if rnn_model_path != None:
            self.policy.load_weights(rnn_model_path)
            logger.log("Load rnn model successfully....")

        if q_net_model_path !=None:
            self.q_net_policy.load_weights(q_net_model_path)
            logger.log("Load q network model successfully ....")

        avg_ppo_rewards = 0.0
        avg_random_rewards = 0.0
        avg_always_migrate_rewards = 0.0
        avg_optimal_rewards = 0.0
        avg_no_migration_rewards = 0.0
        avg_ppo_sample_rewards = 0.0
        avg_drqn_rewards = 0.0

        iter_number = 3
        for i in range(iter_number):
            drqn_rewards_collects, _ = self.q_net_eval_sampler.obtain_samples(is_rnn=rnn_policy)
            sample_reward_collects, _ = self.eval_sampler.obtain_samples(is_rnn=rnn_policy, is_greedy_sample=False)
            reward_collects, system_info_collects = self.eval_sampler.obtain_samples(is_rnn=rnn_policy, is_greedy_sample=True)

            drqn_rewards = np.mean(np.sum(drqn_rewards_collects, axis=-1))
            ppo_rewards = np.mean(np.sum(reward_collects, axis=-1))
            ppo_sample_rewards = np.mean(np.sum(sample_reward_collects, axis=-1))

            random_rewards = random_solution(self.eval_sampler.env, system_info_collects)
            always_migrate_rewards = always_migration_solution(self.eval_sampler.env, system_info_collects)
            optimal_rewards = optimal_solution_for_batch_system_infos(self.eval_sampler.env, system_info_collects)
            no_migration_rewards = no_migration_solution(self.eval_sampler.env, system_info_collects)

            logger.log("---- round "+str(i)+" ----")
            logger.log("eval sample reward", ppo_sample_rewards)
            logger.log("eval drqn reward", drqn_rewards)
            logger.log("eval reward", ppo_rewards)
            logger.log("eval random reward", -random_rewards)
            logger.log("eval always migration reward", -always_migrate_rewards)
            logger.log("eval optimal reward", -optimal_rewards)
            logger.log("no_migration_solution", -no_migration_rewards)

            avg_ppo_rewards += ppo_rewards
            avg_drqn_rewards += drqn_rewards
            avg_random_rewards += random_rewards
            avg_always_migrate_rewards += always_migrate_rewards
            avg_optimal_rewards += optimal_rewards
            avg_no_migration_rewards += no_migration_rewards
            avg_ppo_sample_rewards += ppo_sample_rewards


        logger.logkv("eval sample reward", avg_ppo_sample_rewards / iter_number)
        logger.logkv("eval reward", avg_ppo_rewards / iter_number)
        logger.logkv("eval drqn reward", avg_drqn_rewards/ iter_number)
        logger.logkv("eval random reward", -(avg_random_rewards / iter_number))
        logger.logkv("eval always migration reward", -(avg_always_migrate_rewards / iter_number))
        logger.logkv("eval optimal reward", -(avg_optimal_rewards / iter_number))
        logger.logkv("eval no migration reward", -(avg_no_migration_rewards / iter_number))

        logger.dumpkvs()

if __name__ == "__main__":
    from environment.batch_migration_env import EnvironmentParameters
    from environment.batch_migration_env import BatchMigrationEnv
    from policies.rnn_q_network import RNNQNetwork
    from policies.rnn_policy_with_action_input import RNNPolicyWithValue

    logger.configure(dir="./log/test_with_trace_number_11", format_strs=['stdout', 'log', 'csv'])

    possion_rate_vector = [7, 10, 8, 14, 15, 6, 20, 18, 11, 17, 20, 9, 8, 14, 9, 15, 8, 17, 9, 9, 10, 7, 17, 10,
                           13, 12, 5, 8, 10, 13, 19, 15, 10, 9, 10, 18, 12, 13, 5, 11, 7, 8, 8, 19, 15, 15, 6, 10,
                           5, 20, 17, 5, 5, 16, 5, 19, 19, 19, 9, 20, 17, 14, 17, 17]

    #start point (41.856, 12.442), end point (41.928,12.5387), a region in Roman, Italy.
    env_eval_parameters = EnvironmentParameters(trace_start_index=120,
                                                num_traces=10,
                                                num_base_station=64, optical_fiber_trans_rate=60.0,
                                                server_poisson_rate=possion_rate_vector, client_poisson_rate=4,
                                                server_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                server_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                ratio_lower_bound=200.0,
                                                client_task_data_lower_bound=(0.5 * 1024.0 * 1024.0),
                                                client_task_data_higher_bound=(5 * 1024.0 * 1024.0),
                                                ratio_higher_bound=1000.0, map_width=8000.0, map_height=8000.0,
                                                num_horizon_servers=8, num_vertical_servers=8,
                                                traces_file_path='./environment/rome_traces_coordinate.txt',
                                                transmission_rates=[20.0, 16.0, 12.0, 8.0, 4.0],
                                                trace_length=100,
                                                trace_interval=10,
                                                is_full_observation=False,
                                                is_full_action=True)

    eval_env = BatchMigrationEnv(env_eval_parameters)


    rnn_policy = RNNPolicyWithValue(observation_dim=eval_env._state_dim,
                                    action_dim=eval_env._action_dim,
                                    rnn_parameter=256,
                                    embbeding_size=2)

    q_network = RNNQNetwork(observation_dim=eval_env._state_dim,
                            action_dim=eval_env._action_dim,
                            rnn_parameter=256,
                            fc_parameters=128,
                            epsilon=0.1)

    q_net_sampler = EvaluationSamplerForDRQN(eval_env,
                                            policy=q_network,
                                            batch_size=10,
                                            max_path_length=100)

    eval_sampler = EvaluationSampler(eval_env,
                                     policy=rnn_policy,
                                     batch_size=10,
                                     max_path_length=100)

    tester = Tester(testing_env=eval_env,
                    policy=rnn_policy,
                    q_net_policy = q_network,
                    eval_sampler=eval_sampler,
                    q_net_eval_sampler=q_net_sampler)

    # tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_0",
    #                 q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_0",
    #                 rnn_policy=True)
    # tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_10",
    #                 q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_400",
    #                 rnn_policy=True)
    # tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_20",
    #                 q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_800",
    #                 rnn_policy=True)
    # tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_30",
    #                 q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_1200",
    #                 rnn_policy=True)
    # tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_40",
    #                 q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_1400",
    #                 rnn_policy=True)

    eval_env.ratio_lower_bound = 200.0
    eval_env.ratio_higher_bound = 1000.0
    tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_50",
                    q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_1800",
                    rnn_policy=True)

    eval_env.ratio_lower_bound = 1000.0
    eval_env.ratio_higher_bound = 3000.0
    tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_50",
                    q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_1800",
                    rnn_policy=True)

    eval_env.ratio_lower_bound = 3000.0
    eval_env.ratio_higher_bound = 5000.0
    tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_50",
                    q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_1800",
                    rnn_policy=True)

    eval_env.ratio_lower_bound = 5000.0
    eval_env.ratio_higher_bound = 7000.0
    tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_50",
                    q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_1800",
                    rnn_policy=True)

    eval_env.ratio_lower_bound = 7000.0
    eval_env.ratio_higher_bound = 9000.0
    tester.run_test(rnn_model_path="./checkpoints_ppo_task_embedding/model_checkpoint_epoch_50",
                    q_net_model_path="./q_net_checkpoints/model_checkpoint_gradients_1800",
                    rnn_policy=True)


    #tester.run_test("./checkpoints_ppo_task_embedding/model_checkpoint_epoch_60", rnn_policy=True)
    # tester.run_test("./checkpoints_ppo_task_embedding/model_checkpoint_epoch_70", rnn_policy=True)
    # tester.run_test("./checkpoints_ppo_task_embedding/model_checkpoint_epoch_80", rnn_policy=True)
    # tester.run_test("./checkpoints_ppo_task_embedding/model_checkpoint_epoch_90", rnn_policy=True)

    # tester.run_test("./checkpoints/modle_checkpoint_epoch_20", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_30", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_40", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_50", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_60", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_70", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_80", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_90", rnn_policy=True)
    # tester.run_test("./checkpoints/modle_checkpoint_epoch_100", rnn_policy=True)