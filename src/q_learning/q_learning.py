from typing import Dict

import tensorflow as tf

from src.q_learning.network.dqn import DQN


class QLearning:
    def __init__(self, env, num_actions, agents: Dict, tf_log_dir, file_writer):
        self.env = env
        self.num_actions = num_actions
        self.agents = agents
        self.trainable_agents = [agent for agent in self.agents.values() if agent.trainable]

        self.gamma = 0.9
        self.learning_rate = 0.01
        self.batch_size = 64

        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        self.target_q_network_update_interval = 250
        self.log_interval = 500

        self.tf_log_dir = tf_log_dir
        self.file_writer = file_writer

    def collect_episode(self, num_episodes):
        cur_episode_num = 0

        prev_step_info = {}
        self.env.reset()
        for agent_name in self.env.agents:
            prev_step_info[agent_name] = {}

        while cur_episode_num < num_episodes:
            self.env.reset()

            for _ in range(self.env.num_agents):
                agent_name = self.env.agent_selection
                agent = self.agents[agent_name]
                observation_with_action_mask = self.env.observe(agent_name)
                observation = observation_with_action_mask["observation"]
                action_mask = observation_with_action_mask["action_mask"]
                action = agent.behavior_policy.get_action(tf.expand_dims(tf.identity(observation), 0),
                                                          action_mask).numpy()

                prev_step_info[agent_name]["observation"] = observation
                prev_step_info[agent_name]["action"] = action

                self.env.step(action)

            for agent_name in self.env.agent_iter():
                agent = self.agents[agent_name]

                prev_observation = prev_step_info[agent_name]["observation"]
                action_taken = prev_step_info[agent_name]["action"]

                observation_with_action_mask, reward, done, _ = self.env.last(observe=True)
                observation = observation_with_action_mask["observation"]
                action_mask = observation_with_action_mask["action_mask"]

                if agent.trainable:
                    agent.transition_table.add(s=prev_observation, a=action_taken, r=reward, s2=observation,
                                               is_term=done)

                action = agent.behavior_policy.get_action(tf.expand_dims(tf.identity(observation), 0),
                                                          action_mask).numpy() if not done else None

                prev_step_info[agent_name]["observation"] = observation
                prev_step_info[agent_name]["action"] = action

                self.env.step(action)

            cur_episode_num += 1

    def train_one_step(self, batch_size=8):
        for agent in self.trainable_agents:
            with tf.GradientTape() as tape:
                loss = self.compute_loss(agent, batch_size)
            self.optimizer.minimize(loss=loss, var_list=agent.target_q_network.trainable_weights, tape=tape)

    def compute_loss(self, agent, batch_size):
        s, a, r, s2, term = agent.transition_table.sample(size=batch_size)
        q_value_for_s2 = tf.reduce_max(agent.q_network(s2), axis=1)
        term = (term - 1) * (-1)
        target_q_values = tf.cast(r, dtype=tf.float32) + term * q_value_for_s2 * self.gamma
        predicted_q_values_for_all_actions = agent.target_q_network(s)
        one_hot_actions = tf.one_hot(indices=a, depth=self.num_actions)
        predicted_q_values = tf.reduce_sum(tf.multiply(predicted_q_values_for_all_actions, one_hot_actions), axis=1)
        loss = tf.losses.MSE(y_true=target_q_values, y_pred=predicted_q_values)
        return loss

    def train(self, num_iterations):
        self.collect_episode(100)
        step = 0

        while step <= num_iterations:
            step += 1
            self.collect_episode(1)
            self.train_one_step(batch_size=self.batch_size)

            if step % self.target_q_network_update_interval == 0:
                for agent in self.trainable_agents:
                    agent.q_network = DQN.clone(agent.target_q_network)
                    print(f"[Agent: {agent.name}] [Step: {step}] q_network updated.")

            if step % self.log_interval == 0:
                with self.file_writer.as_default():
                    for agent in self.trainable_agents:
                        loss = self.compute_loss(agent, 5)
                        tf.summary.scalar(f"[{agent.name}] Loss", loss, step=step)
                        tf.summary.scalar(f"[{agent.name}] Epsilon", agent.behavior_policy.epsilon, step=step)
                        tf.summary.flush()

    def evaluate(self, num_episodes=5):
        cur_episode_num = 0
        total_return = {}
        for agent in self.agents.values():
            total_return[agent.name] = 0

        while cur_episode_num < num_episodes:
            self.env.reset()

            episode_return = {}
            for agent in self.agents.values():
                episode_return[agent.name] = 0

            for agent_name in self.env.agent_iter():
                agent = self.agents[agent_name]

                observation_with_action_mask, reward, done, _ = self.env.last(observe=True)
                episode_return[agent_name] += reward
                observation = observation_with_action_mask["observation"]
                action_mask = observation_with_action_mask["action_mask"]

                action = agent.target_policy.get_action(tf.expand_dims(tf.identity(observation), 0),
                                                        action_mask).numpy() if not done else None

                self.env.step(action)

            for agent in self.agents.values():
                total_return[agent.name] += episode_return[agent.name]

            cur_episode_num += 1

        average_return_per_episode = {}
        for agent in self.agents.values():
            average_return_per_episode[agent.name] = total_return[agent.name] / float(num_episodes)

        return average_return_per_episode
