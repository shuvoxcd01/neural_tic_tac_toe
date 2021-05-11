import gym

from src.q_learning.network.dqn import DQN
from src.q_learning.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from src.q_learning.policies.greedy_policy import GreedyPolicy
import tensorflow as tf

from src.q_learning.transition_table.transition_table import TransitionTable


class QLearning:
    def __init__(self, env: gym.Env, input_shape, num_actions):
        self.env = env
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.transition_table = TransitionTable()
        self.q_network = DQN.get_q_network(input_shape=self.input_shape, num_actions=self.num_actions)
        self.target_q_network = DQN.clone(self.q_network)

        self.behavior_policy = EpsilonGreedyPolicy(q_network=self.q_network)
        self.target_policy = GreedyPolicy(q_network=self.target_q_network)

        self.gamma = 0.9
        self.learning_rate = 0.01
        self.batch_size = 64

        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        self.target_q_network_update_interval = 100
        self.eval_interval = 50
        self.log_interval = 20

    def collect_episode(self, num_episodes):
        cur_episode_num = 0

        while cur_episode_num < num_episodes:
            observation = self.env.reset()
            done = False

            while not done:
                action = self.behavior_policy.get_action(tf.expand_dims(tf.identity(observation), 0))
                next_observation, reward, done, _ = self.env.step(action.numpy())
                self.transition_table.add(s=observation, a=action, r=reward, s2=next_observation, is_term=done)
                observation = next_observation

            cur_episode_num += 1

    def train_one_step(self, batch_size=8):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(batch_size)
        self.optimizer.minimize(loss=loss, var_list=self.target_q_network.trainable_weights, tape=tape)

    def compute_loss(self, batch_size):
        s, a, r, s2, term = self.transition_table.sample(size=batch_size)
        q_value_for_s2 = tf.reduce_max(self.q_network(s2), axis=1)
        term = (term - 1) * (-1)
        target_q_values = r + term * q_value_for_s2 * self.gamma
        predicted_q_values_for_all_actions = self.target_q_network(s)
        one_hot_actions = tf.one_hot(indices=a, depth=self.num_actions)
        predicted_q_values = tf.reduce_sum(tf.multiply(predicted_q_values_for_all_actions, one_hot_actions), axis=1)
        loss = tf.losses.MSE(y_true=target_q_values, y_pred=predicted_q_values)
        return loss

    def train(self, num_iterations):

        self.collect_episode(100)

        step = 0

        while step < num_iterations:
            self.collect_episode(1)
            self.train_one_step(batch_size=self.batch_size)

            if step % self.target_q_network_update_interval == 0:
                self.q_network = DQN.clone(self.target_q_network)
                print("q_network updated.")

            if step % self.log_interval == 0:
                print("Step: ", step)
                print("Loss: ", self.compute_loss(5))
                print("epsilon: ", self.behavior_policy.epsilon)

            if step % self.eval_interval == 0:
                print("Average return per episode: ", self.evaluate())

            step += 1

    def evaluate(self, num_episodes=5):
        cur_episode_num = 0
        total_return = 0
        while cur_episode_num < num_episodes:
            observation = self.env.reset()
            done = False
            episode_return = 0

            while not done:
                observation = tf.expand_dims(input=observation, axis=0)
                action = self.target_policy.get_action(observation)
                observation, reward, done, info = self.env.step(action.numpy())

                episode_return += reward

            total_return += episode_return
            cur_episode_num += 1

        average_return_per_episode = total_return / float(num_episodes)

        return average_return_per_episode
