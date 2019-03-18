import random
import tempfile
from collections import deque

import gym
import numpy as np
from gym import wrappers
from keras.layers import Dense
from keras.models import Sequential, model_from_config
from keras.optimizers import Adam


def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


class DDQLAgent:
    def __init__(self, num_states, num_actions, lr=0.0001, eps_min=0.01,
                 epsilon=1, epsilon_decay=0.9993, gamma=0.99, model=None,
                 epochs=1, verbose=0, batch_size=32, memory_limit=5000, env=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = epsilon
        self.eps_min = eps_min
        self.eps_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = lr
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_limit)
        self.model = model
        self.target_model = clone_model(model)
        self.env = env

    def train(self, episodes):
        return self._run(episodes, training=True)

    def test(self, episodes):
        return self._run(episodes)

    def _run(self, episodes, training=False):
        reward_avg = deque(maxlen=100)
        all_rewards = []
        epsilon_vals = []

        for e in range(episodes):
            episode_reward = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.num_states])

            for time in range(1000):
                a = self.select_action(state)
                new_state, r, done, info = self.env.step(a)
                episode_reward += r
                new_state = np.reshape(new_state, [1, self.num_states])
                if training:
                    self.store_memory(state, a, r, new_state, done)

                state = new_state
                if training:
                    if len(self.memory) > self.batch_size:
                        self.replay_experience()
                if done:
                    if training:
                        self.target_model_update()
                    break

            if self.eps > self.eps_min:
                self.eps *= self.eps_decay
            epsilon_vals.append(self.eps)

            reward_avg.append(episode_reward)
            all_rewards.append(episode_reward)
            self.log_episode(e, episode_reward, reward_avg, time)

        self.save_weights()
        return all_rewards, epsilon_vals, reward_avg

    def select_action(self, s):
        if np.random.rand() <= self.eps:
            return np.random.choice(self.num_actions)
        q = self.model.predict(s)
        return np.argmax(q[0])

    def save_weights(self, name='weights'):
        self.model.save_weights('{}.h5'.format(name))

    def load_weights(self, name='weights'):
        self.model.load_weights('{}.h5'.format(name))

    def store_memory(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def target_model_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay_experience(self):
        minibatch = random.sample(self.memory, self.batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        if len(not_done_indices[0]) > 0:
            predict_next_state = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_next_state_target = self.target_model.predict(
                np.vstack(minibatch[:, 3]))

            y[not_done_indices] += np.multiply(self.gamma,
                                               predict_next_state_target[not_done_indices,
                                                                         np.argmax(predict_next_state[not_done_indices, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(self.batch_size), actions] = y
        self.model.fit(
            np.vstack(minibatch[:, 0]), y_target, epochs=self.epochs, verbose=self.verbose)


    def log_episode(self, episode, episode_reward, reward_avg, time):
        print('episode: ', episode, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % np.average(reward_avg),
          ' frames: ', time, ' epsilon: ', '%.2f' % self.eps)
