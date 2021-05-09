import random

from src.q_learning.policies.policy import Policy
import tensorflow as tf


class EpsilonGreedyPolicy(Policy):
    def __init__(self, q_network, epsilon_start, epsilon_end=0.1, epsilon_endt=10000):
        self.q_network = q_network
        self.epsilon = epsilon_start
        self.num_moves = self.q_network.output_shape[-1]

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_endt = epsilon_endt
        self.epsilon_decay_rate = (self.epsilon_end / self.epsilon_start) ** (1.0 / self.epsilon_endt)

    def get_action(self, observation):
        if random.random() <= self.epsilon:
            action = tf.constant(random.randrange(self.num_moves))
        else:
            actions = self.q_network(observation)
            action = tf.math.argmax(actions[0], axis=0)

        self.update_epsilon()

        return action

    def update_epsilon(self):
        self.epsilon = self.epsilon_end if self.epsilon <= self.epsilon_end else self.epsilon ** self.epsilon_decay_rate
