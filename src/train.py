from pettingzoo.classic import tictactoe_v3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
from src.q_learning.agent.agent import Agent
from src.q_learning.network.dqn import DQN
from src.q_learning.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from src.q_learning.policies.greedy_policy import GreedyPolicy
from src.q_learning.q_learning import QLearning
from src.q_learning.transition_table.transition_table import TransitionTable

env = tictactoe_v3.env()
env.reset()

input_shape = (3, 3, 2)
num_actions = 9

agents = {}

for agent_name in env.agents:
    q_network = DQN.get_q_network(input_shape=input_shape, num_actions=num_actions)
    target_q_network = DQN.clone(q_network)
    transition_table = TransitionTable()
    behavior_policy = EpsilonGreedyPolicy(q_network=q_network)
    target_policy = GreedyPolicy(q_network=target_q_network)
    agent = Agent(q_network=q_network, target_q_network=target_q_network, transition_table=transition_table,
                  behavior_policy=behavior_policy, target_policy=target_policy, agent_name=agent_name)

    agents[agent_name] = agent

env.reset()

q_learning = QLearning(env=env, num_actions=num_actions, agents=agents)

q_learning.train(1000000)
