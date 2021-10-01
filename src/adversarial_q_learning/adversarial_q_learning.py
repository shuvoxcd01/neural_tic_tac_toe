import os
from datetime import datetime

import tensorflow as tf

from src.adversarial_q_learning.agent.adversarial_q_learning_agent import AdversarialQLearningAgent
from src.adversarial_q_learning.agent.agent import Agent
from src.adversarial_q_learning.logs import tf_log_parent_dir
from src.adversarial_q_learning.network.dqn import DQN
from src.adversarial_q_learning.policies.random_policy import RandomPolicy
from src.adversarial_q_learning.q_learning import QLearning
from src.adversarial_q_learning.saved_models import saved_model_parent_dir


class AdversarialQLearning:
    def __init__(self, env, num_actions: int, agent_1: AdversarialQLearningAgent, agent_2: AdversarialQLearningAgent):
        self.env = env
        self.num_actions = num_actions
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        self.eval_interval = 1

        date_time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir_to_save_models = os.path.join(saved_model_parent_dir, date_time_now, "adversarial_training")
        self.tf_log_dir = os.path.join(tf_log_parent_dir, date_time_now)
        self.file_writer = tf.summary.create_file_writer(logdir=self.tf_log_dir)

        self.model_saving_interval = 10

    def train_adversarial(self, num_iterations: int):
        step = 0

        while step <= num_iterations:
            step += 1

            self.reset_agents()

            agent_1_copy = self.agent_1.clone(set_trainable=False)
            agent_2_copy = self.agent_2.clone(set_trainable=False)

            adversarial_agent_pair_1 = {self.agent_1.name: self.agent_1, agent_2_copy.name: agent_2_copy}
            adversarial_agent_pair_2 = {agent_1_copy.name: agent_1_copy, self.agent_2.name: self.agent_2}

            adversarial_agent_pairs = [adversarial_agent_pair_1, adversarial_agent_pair_2]

            for adversarial_agent_pair in adversarial_agent_pairs:
                q_learning = QLearning(env=self.env, num_actions=self.num_actions, agents=adversarial_agent_pair,
                                       tf_log_dir=self.tf_log_dir, file_writer=self.file_writer)
                q_learning.train()

            agents = [self.agent_1, self.agent_2]

            if step % self.eval_interval == 0:
                self.evaluate_agents(agents, step)

            if step % self.model_saving_interval == 0:
                self.save_target_networks(agents, step)

        for agent in [self.agent_1, self.agent_2]:
            DQN.save_model(model=agent.target_q_network, saved_model_dir=self.dir_to_save_models,
                           saved_model_name=agent.name + "_" + str(step))

    def save_target_networks(self, agents, step):
        for agent in agents:
            DQN.save_model(model=agent.target_q_network, saved_model_dir=self.dir_to_save_models,
                           saved_model_name=agent.name + "_" + str(step))
            print(f"[{agent.name}] Step {step}: Target network saved.")

    def evaluate_agents(self, agents, step):
        with self.file_writer.as_default():
            avg_return_per_ep = self.evaluate(agents)
            for agent in agents:
                agent_name = agent.name
                tf.summary.scalar(f"[{agent_name}] Average return per episode against trained adversary",
                                  avg_return_per_ep[agent_name], step=step)
                tf.summary.flush()

            agent_1_avg_ret_against_random = self.evaluate_against_random_agent(agent=self.agent_1,
                                                                                agent_moves_first=True)
            tf.summary.scalar(f"[{self.agent_1.name}] Average return per episode against random agent",
                              agent_1_avg_ret_against_random[self.agent_1.name], step=step)
            tf.summary.flush()

            agent_2_avg_ret_against_random = self.evaluate_against_random_agent(agent=self.agent_2,
                                                                                agent_moves_first=False)

            tf.summary.scalar(f"[{self.agent_2.name}] Average return per episode against random agent",
                              agent_2_avg_ret_against_random[self.agent_2.name], step=step)
            tf.summary.flush()

    def reset_agents(self):
        self.agent_1.clear_transition_table()
        self.agent_2.clear_transition_table()
        self.agent_1.reset_behavior_policy()
        self.agent_2.reset_behavior_policy()

    def evaluate(self, agents, num_episodes=10):
        cur_episode_num = 0
        total_return = {}
        for agent in agents:
            total_return[agent.name] = 0

        while cur_episode_num < num_episodes:
            self.env.reset()
            selected_agent_index = 0

            episode_return = {}
            for agent in agents:
                episode_return[agent.name] = 0

            for _ in self.env.agent_iter():
                agent = agents[selected_agent_index]

                observation_with_action_mask, reward, done, _ = self.env.last(observe=True)
                episode_return[agent.name] += reward
                observation = observation_with_action_mask["observation"]
                action_mask = observation_with_action_mask["action_mask"]

                action = agent.target_policy.get_action(tf.expand_dims(tf.identity(observation), 0),
                                                        action_mask).numpy() if not done else None

                self.env.step(action)

                selected_agent_index = 1 - selected_agent_index

            for agent in agents:
                total_return[agent.name] += episode_return[agent.name]

            cur_episode_num += 1

        average_return_per_episode = {}
        for agent in agents:
            average_return_per_episode[agent.name] = total_return[agent.name] / float(num_episodes)

        return average_return_per_episode

    def evaluate_against_random_agent(self, agent: Agent, agent_moves_first: bool):
        random_policy = RandomPolicy(num_actions=self.num_actions)
        random_agent = Agent(q_network=None, target_q_network=None, transition_table=None, behavior_policy=None,
                             target_policy=random_policy, agent_name="Random_Player", trainable=False)

        agents = [random_agent, agent]
        if agent_moves_first:
            agents = [agent, random_agent]

        average_return_per_episode = self.evaluate(agents)

        return average_return_per_episode
