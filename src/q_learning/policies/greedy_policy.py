import tensorflow as tf

from src.q_learning.policies.policy import Policy


class GreedyPolicy(Policy):
    def __init__(self, q_network):
        self.q_network = q_network

    def get_action(self, observation, action_mask):
        actions = self.q_network(observation)[0]

        actions = tf.nn.softmax(actions) * action_mask
        best_action = tf.math.argmax(actions, axis=0)
        best_action = tf.cast(best_action, dtype=tf.int32)

        return best_action
