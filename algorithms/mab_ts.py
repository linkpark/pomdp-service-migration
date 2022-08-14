import numpy as np

class MABTSServiceMigration(object):
    def __init__(self,
                 env):
        self.env = env
        self.context_vector = [np.identity(2*env._num_base_station + 1)] * env._num_base_station
        # mu_i
        self.estimate_feature_vector = [np.zeros(2*env._num_base_station + 1)] * env._num_base_station
        # f_i
        self.contextual_service_cost = [np.zeros(2*env._num_base_station + 1)] * env._num_base_station
        self.v = 0.1

        self.sum_of_rewards = np.zeros(env._num_base_station)
        self.visit_time = np.zeros(env._num_base_station)

    def one_hot(self, position):
        ret = np.zeros(env._num_base_station)
        ret[position] = 1.0

        return ret


    def step(self, ob):
        thetas = []
        location_index = int(ob[0])
        service_index = int(ob[1])
        request_load = ob[3]

        b = np.concatenate([[1.0], self.one_hot(location_index), self.one_hot(service_index)])

        #print("b shape: ", b.shape)
        for i in range(self.env._num_base_station):
            b_transpose = np.transpose(b)

            guassian_mean = np.dot(b_transpose,
                                   self.estimate_feature_vector[i])

            B_inverse = np.linalg.inv(self.context_vector[i])
            guassian_std = np.dot(b_transpose, B_inverse)
            guassian_std = (self.v**2) * np.dot(guassian_std, b)

            theta = np.random.normal(guassian_mean, guassian_std)
            thetas.append(theta)

        # print("guassian_mean: ", guassian_mean)
        # print("guassian_std: ", guassian_std)

        action = np.argmin(thetas)

        # print("actions: ", action)

        ob, reward, done, _ = self.env.step(action)
        # print("observations: ", ob)
        reward = -reward

        # print("np.dot(b,b_transpose)", np.dot(b, b_transpose).shape)
        # update
        # print("b: ", b)
        # print("np.dot: ", np.dot(b, reward))
        self.context_vector[action] = self.context_vector[action] + np.dot(b, b_transpose)
        self.contextual_service_cost[action] = self.contextual_service_cost[action] + b * reward
        self.estimate_feature_vector[action] = np.dot(np.linalg.inv(self.context_vector[i]) , self.contextual_service_cost[action])

        # print("self.context_vector[action] shape: ", self.context_vector[action].shape)
        # print("self.contextual_service_cost[action]", self.contextual_service_cost[action].shape)
        # print("self.estimate_feature_vector[action]", self.estimate_feature_vector[action].shape)

        return reward, ob

    def mab_step_non_batch(self, ob):
        q_values = []
        for j in range(self.env._num_base_station):
            if self.visit_time[j] == 0:
                q_values.append(0)
            else:
                q_values.append(self.sum_of_rewards[j] / self.visit_time[j])

        action = np.argmin(q_values)

        ob, rewards, false, done = env.step(action)

        rewards = - rewards

        self.sum_of_rewards[action] += rewards

        self.visit_time[action] += 1.0
        return rewards, ob

    def mab_step_batch(self, ob):
        actions = []
        for i in range(len(ob)):
            q_values = []
            for j in range(self.env._num_base_station):
                if self.visit_time[i][j] == 0:
                    q_values.append(0)
                else:
                    q_values.append(self.sum_of_rewards[i][j] / self.visit_time[i][j])

            action = np.argmin(q_values)
            actions.append(action)

        ob, rewards, false, done = env.step(actions)



        for i in range(len(rewards)):
            action = actions[i]
            self.sum_of_rewards[i][action] += rewards[i]
            self.visit_time[i][action] += 1.0

        return rewards, ob


    def train(self, num_iteration, time_slot=100):
        total_rewards = []
        for i in range(num_iteration):
            ob = self.env.reset()

            total_reward = 0.0
            for j in range(time_slot):
                reward, ob = self.mab_step_non_batch(ob)
                reward = np.array(reward)
                total_reward += reward

            print("episodic reward: ", np.mean(total_reward))
            total_rewards.append(np.mean(total_reward))

        return total_rewards


class MABTSGuassianServiceMigration(object):
    def __init__(self,
                 env):
        self.env = env
        self.std = np.ones((self.env.batch_size, env._num_base_station))
        self.mean = np.zeros((self.env.batch_size, env._num_base_station))

    def step(self, ob):
        actions = []
        for i in range(len(ob)):
            thetas = []
            for j in range(self.env._num_base_station):
                theta = np.random.normal(self.mean[i][j], self.std[i][j])
                thetas.append(theta)

            # print("guassian_mean: ", guassian_mean)
            # print("guassian_std: ", guassian_std)

            action = np.argmin(thetas)
            actions.append(action)
        # print("actions: ", action)

        ob, rewards, done, _ = self.env.step(actions)
        # print("observations: ", ob)

        for i in range(self.env.batch_size):
            reward = -rewards[i]
            action = actions[i]

            return_std = 0.1
            self.mean[i][action] = (return_std**2 * self.mean[i][action] + reward * self.std[i][action]**2) / (return_std**2 + self.std[i][action]**2)
            self.std[i][action] = np.sqrt(1.0 / (return_std**-2 + self.std[i][action]**-2))

        return rewards, ob


    def train(self, num_iteration, time_slot=100):
        total_rewards = []
        for i in range(num_iteration):
            ob = self.env.reset()

            total_reward = np.zeros(self.env.batch_size)
            for j in range(time_slot):
                reward, ob = self.step(ob)
                reward = np.array(reward)
                total_reward += reward

            print("episodic reward: ", np.mean(total_reward))
            total_rewards.append(np.mean(total_reward))

        return total_rewards
