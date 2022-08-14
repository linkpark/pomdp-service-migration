# This class is responsible for the training process of the PPO algorithm
import tensorflow as tf
import numpy as np
import time
from utils import logger
from policies.random_migrate_policy import RandomMigratePolicy
from policies.always_migrate_policy import AlwaysMigratePolicy

from policies.random_solution import random_solution
from policies.always_migration_solution import always_migration_solution
from policies.optimal_solution import optimal_solution_for_batch_system_infos
from policies.no_migration_solution import no_migration_solution
from utils.logger import Logger

class Trainer(object):
    def __init__(self,
                 train_env,
                 eval_env,
                 algo,
                 sampler,
                 sample_processor,
                 update_batch_size,
                 policy,
                 n_itr,
                 save_interval,
                 save_path,
                 test_interval=0,
                 eval_sampler=None):
        self.train_env = train_env
        self.eval_env = eval_env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.save_interval = save_interval
        self.update_batch_size = update_batch_size
        self.algo = algo
        self.eval_sampler = eval_sampler
        self.test_interval = test_interval
        self.save_path = save_path

    def train(self, rnn_policy=False, is_test=True, is_save =True, is_log=True,
              std_reward=0.0,
              mean_reward=0.0,
              avg_random_rewards=0.0,
              avg_always_migrate_rewards=0.0,
              optimal_migrate_rewards=0.0,
              no_migrate_rewards=0.0):
        avg_ret = []
        avg_loss = []
        avg_latencies = []

        for itr in range(self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling trajectories from environment ...")
            start_time = time.time()

            paths = self.sampler.obtain_samples(is_rnn=rnn_policy, log=False, log_prefix='',reward_mean=mean_reward,
                                                reward_std=std_reward)

            end_time = time.time()

            logger.log("Sampling spend time: ", (end_time - start_time), "s")
            logger.log("Processing trajectories ...")
            start_time = time.time()
            samples_data = self.sample_processor.process_samples(paths)
            end_time = time.time()
            logger.log("Processing spend time: ", (end_time - start_time), "s")

            start_time = time.time()
            avg_random_rewards = random_solution(self.train_env, samples_data["system_info"])
            avg_always_migrate_rewards = always_migration_solution(self.train_env, samples_data["system_info"])
            no_migrate_rewards =no_migration_solution(self.train_env, samples_data["system_info"])
            end_time = time.time()

            logger.log("Baselien algorithms: ", (end_time - start_time), "s")
            # update ppo target

            logger.log("Updating policies ....")
            start_time = time.time()
            policy_losses, ent_losses, value_losses = self.algo.update_dracm(samples_data, self.update_batch_size)
            end_time = time.time()

            logger.log("Update spend time: ", (end_time - start_time), "s")

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['un_norm_rewards'], axis=-1)
            avg_reward = np.mean(ret)

            if is_log:
                logger.logkv("Itr", itr)
                logger.logkv("policy loss: ", np.round(np.mean(policy_losses), 2))
                logger.logkv("value loss: ", np.round(np.mean(value_losses), 2))
                logger.logkv("entropy loss: ", np.round(np.mean(ent_losses), 2))
                logger.logkv("average reward: ", np.round(np.mean(avg_reward), 2))
                logger.logkv("average random reward: ", -np.round(np.mean(avg_random_rewards), 2))
                logger.logkv("average always migrate reward: ", -np.round(np.mean(avg_always_migrate_rewards),2))
                logger.logkv("average never migrate rewards: ", -np.round(np.mean(no_migrate_rewards), 2))
                logger.logkv("optimal migrate reward: ", -np.round(np.mean(optimal_migrate_rewards), 2))

                logger.dumpkvs()

            if itr % self.test_interval == 0 and is_test == True:
                avg_ppo_rewards = 0.0
                avg_random_rewards = 0.0
                avg_always_migrate_rewards = 0.0
                avg_optimal_rewards = 0.0
                avg_no_migration_rewards = 0.0

                num_iter = 4
                for i in range(num_iter):
                    reward_collects, system_info_collects = self.eval_sampler.obtain_samples(is_rnn=rnn_policy)
                    ppo_rewards = np.mean(np.sum(reward_collects, axis=-1))
                    random_rewards = random_solution(self.eval_sampler.env, system_info_collects)
                    always_migrate_rewards = always_migration_solution(self.eval_sampler.env, system_info_collects)
                    #optimal_rewards = optimal_solution_for_batch_system_infos(self.eval_sampler.env, system_info_collects)
                    no_migrate_rewards = no_migration_solution(self.eval_sampler.env, system_info_collects)

                    avg_ppo_rewards += ppo_rewards
                    avg_random_rewards += random_rewards
                    avg_always_migrate_rewards += always_migrate_rewards
                    #avg_optimal_rewards += optimal_rewards
                    avg_no_migration_rewards += no_migrate_rewards
                if is_log:
                    logger.logkv("eval reward", avg_ppo_rewards / num_iter)
                    logger.logkv("eval random reward", -(avg_random_rewards / num_iter))
                    logger.logkv("eval always migration reward", -(avg_always_migrate_rewards / num_iter))
                    logger.logkv("eval optimal reward", -(avg_optimal_rewards / num_iter))
                    logger.logkv("eval no migration reward", -(avg_no_migration_rewards)/ num_iter)

                    logger.dumpkvs()

            if itr % self.save_interval == 0 and is_save == True:
                logger.log("save model weights ... ")
                self.policy.save_weights(self.save_path+str(itr))

