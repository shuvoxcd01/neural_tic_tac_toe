import gym

from src.q_learning.network.dqn import DQN
from src.q_learning.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from src.q_learning.policies.greedy_policy import GreedyPolicy
import tensorflow as tf

from src.q_learning.transition_table.transition_table import TransitionTable


class QLearning:
    def __init__(self, env: gym.Env., input_shape, num_actions):
        self.env = env
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.transition_table = TransitionTable()
        self.q_network = DQN(input_shape=self.input_shape, num_actions=self.num_actions)
        self.target_q_network = tf.keras.models.clone_model(self.q_network)

        self.behavior_policy = EpsilonGreedyPolicy(q_network=self.q_network)
        self.target_policy = GreedyPolicy(q_network=self.target_q_network)

    def collect_episode(self, num_episodes):
        cur_episiode_num = 0

        while cur_episiode_num < num_episodes:
            observation = self.env.reset()
            done = False

            while not done:
                action = self.behavior_policy.get_action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                self.transition_table.add(s=observation, a=action, r=reward, s2=next_observation, is_term=done)
                observation = next_observation

            cur_episiode_num += 1

    def train(self, batch_size=8):
        pass
