import numpy as np
from sklearn.linear_model import LogisticRegression
from contextualbandits.online import BootstrappedTS
from contextualbandits.online import AdaptiveGreedy
from copy import deepcopy


class MABPolicy(object):
    def __init__(self,
                 observation_dim,
                 action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
        self.beta_prior_ts = ((3./action_dim, 4), 2)
        self.is_rnn = False

        self.model = BootstrappedTS(deepcopy(self.base_algorithm), nchoices = action_dim,
                                 beta_prior = self.beta_prior_ts, random_state = 2222)

    def sample(self, observations):
        observations = np.array(observations, dtype=np.float32)
        actions = self.model.predict(observations)

        return actions

    def update(self, observations, actions, rewards):
        self.model.fit(X=observations, a=actions, r=rewards, warm_start=True)
