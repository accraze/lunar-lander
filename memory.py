from collections import deque, namedtuple
import random

import numpy as np


Transition = namedtuple(
    'Transition', 'state_0, action, reward, state_1, terminal_1')


class SequentialMemory(object):
    def __init__(self, limit, window_length, ignore_episode_boundaries=False, **kwargs):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

        self.limit = limit
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.observations = deque(maxlen=limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of transitions
        """
        if batch_idxs is None:
            batch_idxs = self.sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1

        # Create transitions
        transitions = []
        for idx in batch_idxs:
            terminal_0 = self.terminals[idx - 2]
            while terminal_0:
                idx = self.sample_batch_indexes(
                    self.window_length + 1, self.nb_entries, size=1)[0]
                terminal_0 = self.terminals[idx - 2]

            state_0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    break
                state_0.insert(0, self.observations[current_idx])
            while len(state_0) < self.window_length:
                state_0.insert(0, self.zeroed_observation(state_0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal_1 = self.terminals[idx - 1]

            state_1 = [np.copy(x) for x in state_0[1:]]
            state_1.append(self.observations[idx])

            transitions.append(Transition(state_0=state_0, action=action, reward=reward,
                                          state_1=state_1, terminal_1=terminal_1))
        return transitions

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory
        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations
        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory
        # Returns
            Dict of config
        """
        config = {}
        config['window_length'] = self.window_length
        config['ignore_episode_boundaries'] = self.ignore_episode_boundaries
        config['limit'] = self.limit
        return config

    def get_recent_state(self, current_observation):
        """Return list of last observations
        # Argument
            current_observation (object): Last observation
        # Returns
            A list of the last observations
        """
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx -
                                                     1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or \
                    (not self.ignore_episode_boundaries and current_terminal):
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, self.zeroed_observation(state[0]))
        return state

    def sample_batch_indexes(self, low, high, size):
        """Return a sample of (size) unique elements between low and high
        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick
        # Returns
            A list of samples of length size, with values between low and high
        """
        if high - low >= size:
            try:
                r = xrange(low, high)
            except NameError:
                r = range(low, high)
            batch_idxs = random.sample(r, size)
        else:
            batch_idxs = np.random.random_integers(low, high - 1, size=size)
        return batch_idxs

    def zeroed_observation(self, observation):
        """Return an array of zeros with same shape as given observation
        """
        if hasattr(observation, 'shape'):
            return np.zeros(observation.shape)
        elif hasattr(observation, '__iter__'):
            out = []
            for x in observation:
                out.append(self.zeroed_observation(x))
            return out
        else:
            return 0.
