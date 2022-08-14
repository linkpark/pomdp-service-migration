import numpy as np

class RandomMigratePolicy(object):
    def __init__(self,
                 observation_dim,
                 action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def sample(self, observations):
        batch_size = np.array(observations).shape[0]
        actions = np.random.randint(0,self.action_dim,size=[batch_size])

        return actions

