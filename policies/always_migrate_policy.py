import numpy as np

class AlwaysMigratePolicy(object):
    def __init__(self,
                 observation_dim,
                 action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def greedy_sample(self, observations):
        observations = np.array(observations, dtype=np.float32)
        actions = np.squeeze(np.array(observations[:,0],dtype=np.int32))

        return actions

    def sample(self, observations):
        observations = np.array(observations, dtype=np.float32)
        actions = np.squeeze(np.array(observations[:,0],dtype=np.int32))

        return actions