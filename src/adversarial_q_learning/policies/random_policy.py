import random

import numpy as np
import tensorflow as tf

from src.adversarial_q_learning.policies.policy import Policy


class RandomPolicy(Policy):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_action(self, observation, action_mask):
        binary_mask = action_mask.astype(bool)
        valid_actions = np.arange(self.num_actions)[binary_mask]
        action = tf.constant(random.choice(valid_actions))

        return action
