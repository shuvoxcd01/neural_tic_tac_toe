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
        self.q_network = DQN.get_q_network(input_shape=self.input_shape, num_actions=self.num_actions)
        self.target_q_network = tf.keras.models.clone_model(self.q_network)

        DQN.freeze_layers(self.q_network)

        self.behavior_policy = EpsilonGreedyPolicy(q_network=self.q_network)
        self.target_policy = GreedyPolicy(q_network=self.target_q_network)

        self.gamma = 0.9
        self.learning_rate = 0.01

        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        self.target_q_network_update_interval = 100
        self.eval_interval = 500
        self.log_interval = 10

    def collect_episode(self, num_episodes):
        cur_episode_num = 0

        while cur_episode_num < num_episodes:
            observation = self.env.reset()
            done = False

            while not done:
                action = self.behavior_policy.get_action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                self.transition_table.add(s=observation, a=action, r=reward, s2=next_observation, is_term=done)
                observation = next_observation

            cur_episode_num += 1

    def train_one_step(self, batch_size=8):
        loss = self.compute_loss(batch_size)
        self.optimizer.minimize(loss=loss, var_list=self.target_q_network.trainable_weights)

    def compute_loss(self, batch_size):
        s, a, r, s2, term = self.transition_table.sample(size=batch_size)
        q_value_for_s2 = tf.reduce_max(self.q_network(s2))
        term = (term - 1) * (-1)
        target_q_value = r + term * q_value_for_s2 * self.gamma
        predicted_q_value = self.target_q_network(s)
        loss = tf.losses.MSE(y_true=target_q_value, y_pred=predicted_q_value)
        return loss

    def train(self, num_iterations):

        self.collect_episode(1000)

        step = 0

        while step < num_iterations:
            self.collect_episode(10)
            self.train_one_step()

            if step % self.target_q_network_update_interval:
                self.target_q_network = tf.keras.models.clone_model(self.q_network)

            if step % self.log_interval:
                self.compute_loss(10)

            # ToDo evaluation
