from src.q_learning.policies.policy import Policy
import tensorflow as tf


class GreedyPolicy(Policy):
    def __init__(self, q_network):
        self.q_network = q_network

    def get_action(self, observation):
        actions = self.q_network(observation)

        best_action = tf.math.argmax(actions[0], axis=0)
        best_action = tf.cast(best_action, dtype=tf.int32)

        return best_action
