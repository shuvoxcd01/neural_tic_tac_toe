import numpy as np
import tensorflow as tf

from src.adversarial_q_learning.policies.policy import Policy


class HumanPolicy(Policy):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def get_action(self, observation, action_mask):
        actions = np.arange(self.num_actions)[action_mask.astype(bool)]
        action = int(input(f"Select an action from {actions.tolist()}"))

        return tf.constant(action)
