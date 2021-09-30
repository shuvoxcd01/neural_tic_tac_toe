import random

import numpy as np
import tensorflow as tf

from src.q_learning.network.dqn import DQN
from src.q_learning.policies.policy import Policy


class EpsilonGreedyPolicy(Policy):
    def __init__(self, q_network, epsilon_start=1.0, epsilon_end=0.1, epsilon_endt=100000):
        self.q_network = q_network
        self.epsilon = epsilon_start
        self.num_moves = self.q_network.output_shape[-1]

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_endt = epsilon_endt
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_endt

    def get_action(self, observation, action_mask):
        if random.random() <= self.epsilon:
            binary_mask = action_mask.astype(bool)
            valid_actions = np.arange(self.num_moves)[binary_mask]
            action = tf.constant(random.choice(valid_actions))
        else:
            actions = self.q_network(observation)[0]
            actions = tf.nn.softmax(actions) * action_mask
            action = tf.math.argmax(actions, axis=0)
            action = tf.cast(action, dtype=tf.int32)

        self.update_epsilon()

        return action

    def update_epsilon(self):
        self.epsilon = self.epsilon_end if self.epsilon <= self.epsilon_end else (
                self.epsilon - self.epsilon_decay_rate)

    def get_q_network(self):
        return DQN.clone(self.q_network)

    def set_q_network(self, q_network):
        self.q_network = q_network
