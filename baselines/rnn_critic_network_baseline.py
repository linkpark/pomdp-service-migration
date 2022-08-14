from baselines.baseline_base import Baseline

import numpy as np

class RNNCriticNetworkBaseline(Baseline):
    def __init__(self,
                 critic_network,):
        self.critic_network = critic_network

    def get_param_values(self):
        """
        Returns the parameter values of the baseline object

        """
        return self.critic_network.trainable_variables()

    def set_params(self, value):
        """
        Sets the parameter values of the baseline object

        Args:
            value: parameter value to be set

        """
        self.critic_network.set_weights(value)

    def fit(self, paths, target_key='returns'):
        """
        Fits the baseline model with the provided paths

        Args:
            paths: list of paths

        """
        pass

    def predict(self, path):
        """
        Predicts the reward baselines for a provided trajectory / path

        Args:
            path: dict of lists/numpy array containing trajectory / path information
                  such as "observations", "rewards", ...

        Returns: numpy array of the same length as paths["observations"] specifying the reward baseline

        """
        observations = path["observations"]
        actions = path["actions"]

        shift_actions = np.concatenate(([0], actions[0:-1]))

        observations = np.expand_dims(observations, axis=0)
        shift_actions = np.expand_dims(shift_actions, axis=0)

        input = (observations, shift_actions)
        values = np.squeeze(self.critic_network.predict(input).numpy())
        return values

    def log_diagnostics(self, paths, prefix):
        """
        Log extra information per iteration based on the collected paths
        """
        pass