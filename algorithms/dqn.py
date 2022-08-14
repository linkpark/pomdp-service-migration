
import tensorflow as tf
import utils.logger as logger
import numpy as np

from policies.random_solution import random_solution
from policies.always_migration_solution import always_migration_solution

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.math.abs(x) < delta,
        tf.math.square(x) * 0.5,
        delta * (tf.math.abs(x) - 0.5 * delta)
    )

class DQN(object):
    def __init__(self,
                 q_network,
                 replay_buffer,
                 sampler,
                 action_dim,
                 gamma,
                 optimizer,
                 log_interval,
                 eval_sampler,
                 save_interval,
                 sample_interval,
                 model_path = None,
                 ):
        self.q_net = q_network
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.action_dim = action_dim
        self.gamma = gamma
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.eval_sampler = eval_sampler
        self.save_interval = save_interval
        self.sample_interval = sample_interval
        self.model_path = model_path

    def update_q_net(self, obs, rewards, actions, next_max_q_values):
        with tf.GradientTape() as tape:
            q_values = self.q_net(obs)
            q_values_seltected = tf.reduce_sum(q_values * tf.one_hot(actions, depth=self.action_dim, axis=-1), axis=-1)

            q_values_target = rewards + self.gamma * next_max_q_values

            td_error = q_values_seltected - q_values_target
            q_loss = tf.reduce_mean(huber_loss(td_error))

            gradients = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.q_net.trainable_variables))

        return q_loss.numpy()


    def train(self, update_steps, optimal_batch_size):
        sampler_times = update_steps / self.sample_interval
        for i in range(update_steps):
            if i % self.sample_interval == 0:
                n_sample = int(i / self.sample_interval)

                if n_sample < int(sampler_times / 2):
                    epsilon = 1.0 - ( float(n_sample) / float(sampler_times / 2) )* 0.9
                else:
                    epsilon = 0.1
                print("Start sampling........", n_sample, "total sample times: ", sampler_times, "epsilon: ", epsilon)
                paths = self.sampler.obtain_samples(epsilon=epsilon)
                self.replay_buffer.add(paths)

            obs, actions, rewards, max_q_values = self.replay_buffer.sample(optimal_batch_size)
            q_value_loss = self.update_q_net(obs, actions, rewards, max_q_values)

            if i % self.log_interval == 0:
                logger.logkv("training average rewards: ", np.mean(np.sum(rewards, axis=-1)))
                logger.logkv("q_value_loss", q_value_loss)
                reward_collects, system_info_collects = self.eval_sampler.obtain_samples(is_rnn=True)
                mean_rewards = np.mean(np.sum(reward_collects, axis=-1))
                avg_random_rewards = random_solution(self.eval_sampler.env, system_info_collects)
                avg_always_migrate_rewards = always_migration_solution(self.eval_sampler.env, system_info_collects)
                logger.logkv("eval reward", mean_rewards)
                logger.logkv("random reward", avg_random_rewards)
                logger.logkv("always migrate reward", avg_always_migrate_rewards)
                logger.dumpkvs()

            if i % self.save_interval == 0 and self.model_path != None:
                logger.log("save model weights ... ")
                self.q_net.save_weights(self.model_path + "_" + str(i))