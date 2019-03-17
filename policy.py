import numpy as np


class EpsilonGreedyQPolicy(object):
    """Implement the epsilon greedy policy
    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=.1):
        self.eps = eps

    def _set_agent(self, agent):
        self.agent = agent

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            # take random action with probability eps
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy
        # Returns
            Dict of config
        """
        config = {}
        config['eps'] = self.eps
        return config
