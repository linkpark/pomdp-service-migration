import tensorflow as tf
import numpy as np
from utils import logger
import sys


class DRACM(object):
    def __init__(self,
                 policy,
                 value_function,
                 policy_optimizer,
                 value_optimizer,
                 is_rnn=False,
                 is_shared_critic_net = True,
                 num_inner_grad_steps=4,
                 clip_value=0.2,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 entropy_coef = 0.01):
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.num_inner_grad_steps = num_inner_grad_steps
        self.vf_coef = vf_coef
        self.clip_value=clip_value
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_function = value_function
        self.is_rnn = is_rnn
        self.is_shared_critic_net =is_shared_critic_net

    def value_function_pretrain(self, samples, update_steps, batch_size,inter_val):
        if self.value_function == None:
            return

        dataset = tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
        for i in range(update_steps):
            batch_count = tf.constant(0.0)
            vf_loss = tf.constant(0.0)
            for batch in dataset:
                with tf.GradientTape() as tape:
                    if self.is_rnn:
                        pred_values = self.value_function.predict((batch["observations"], batch["shift_actions"]))
                    else:
                        pred_values = self.value_function.predict(batch["observations"])
                    vf_loss_func = 0.5 * tf.math.reduce_mean(tf.math.square(pred_values - batch["returns"]))

                    value_gradients = tape.gradient(vf_loss_func, self.value_function.trainable_variables)

                    self.value_optimizer.apply_gradients(zip(value_gradients, self.value_function.trainable_variables))
                    batch_count += 1.0
                    vf_loss += vf_loss_func

            vf_loss = vf_loss / batch_count
            if i % inter_val == 0:
                logger.log("vale loss is :", vf_loss.numpy())

    def update_dracm(self, samples, batch_size):
        # before update record the old logits
        if self.is_rnn:
            _, old_logits, _ = self.policy(samples["observations"], samples["shift_actions"])

            if self.value_function != None:
                values = self.value_function.predict((samples["observations"], samples["shift_actions"]))
                samples["old_values"] = values
        else:
            _, old_logits, _ = self.policy(samples["observations"])
            if self.value_function != None:
                values = self.value_function.predict(samples["observations"])
                samples["old_values"] = values

        samples["old_logits"] = old_logits

        # using tf.dataset to handle the input data
        dataset = tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)

        policy_loss = tf.constant(0.0)
        value_loss = tf.constant(0.0)
        ent_loss = tf.constant(0.0)

        logger.logkv("batch size", batch_size)
        for i in range(self.num_inner_grad_steps):
            for batch in dataset:
                with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
                    if self.is_rnn:
                        # this update is for pomdp only
                        _, new_logits, _ = self.policy(batch["observations"], batch["shift_actions"])
                    else:
                        _, new_logits, _ = self.policy(batch["observations"])

                    likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(batch["actions"], batch["old_logits"], new_logits)

                    clipped_obj = tf.math.minimum(likelihood_ratio * batch["advantages"],
                                             tf.clip_by_value(likelihood_ratio,
                                                              1.0 - self.clip_value, 1.0 + self.clip_value) * batch["advantages"])

                    surr_obj = -tf.math.reduce_mean(clipped_obj)

                    #calculate the value loss
                    if self.value_function != None:
                        if self.is_rnn:
                            pred_values = self.value_function.predict((batch["observations"], batch["shift_actions"]))
                        else:
                            pred_values = self.value_function.predict(batch["observations"])
                        vpredclipped = batch["old_values"] + tf.clip_by_value(pred_values - batch["old_values"],
                                                                              -self.clip_value, self.clip_value)
                        vf_loss_1 = tf.math.square(pred_values - batch["returns"])
                        vf_loss_2 = tf.math.square(vpredclipped - batch["returns"])
                        vf_loss = 0.5 * tf.math.reduce_mean(tf.math.maximum(vf_loss_1, vf_loss_2))

                    # calculate the entropy loss
                    entropy_loss = tf.math.reduce_mean(self.policy.distribution.entropy_sym(new_logits))

                    if self.is_shared_critic_net:
                        pg_loss = surr_obj - self.entropy_coef * entropy_loss + self.vf_coef * vf_loss
                    else:
                        pg_loss = surr_obj - self.entropy_coef * entropy_loss

                    policy_gradients = policy_tape.gradient(pg_loss, self.policy.trainable_variables)

                    if self.max_grad_norm is not None:
                        policy_gradients, _grad_norm = tf.clip_by_global_norm(policy_gradients, self.max_grad_norm)

                    self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy.trainable_variables))

                    policy_loss += surr_obj.numpy()
                    ent_loss += entropy_loss.numpy()
                    # Total loss policy gradient loss + value loss - entropy_loss (we want to maximise entropy)
                    #total_loss = surr_obj - self.entropy_coef * entropy_loss
                    if self.value_function != None and self.is_shared_critic_net == False:
                        value_gradients = value_tape.gradient(vf_loss, self.value_function.trainable_variables)

                        if self.max_grad_norm is not None:
                            value_gradients, _value_grad_norm = tf.clip_by_global_norm(value_gradients, self.max_grad_norm)

                        self.value_optimizer.apply_gradients(zip(value_gradients, self.value_function.trainable_variables))

                    value_loss += vf_loss.numpy()

        policy_loss = (policy_loss / float(self.num_inner_grad_steps))
        value_loss = (value_loss / float(self.num_inner_grad_steps))
        ent_loss = (ent_loss / float(self.num_inner_grad_steps))

        return policy_loss, ent_loss, value_loss

