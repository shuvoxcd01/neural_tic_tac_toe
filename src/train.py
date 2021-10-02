from pettingzoo.classic import tictactoe_v3

from src.adversarial_q_learning.adversarial_q_learning import AdversarialQLearning
from src.adversarial_q_learning.agent.adversarial_q_learning_agent import AdversarialQLearningAgent
from src.adversarial_q_learning.network.dqn import DQN
from src.adversarial_q_learning.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from src.adversarial_q_learning.policies.greedy_policy import GreedyPolicy
from src.adversarial_q_learning.transition_table.transition_table import TransitionTable

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

env = tictactoe_v3.env()
env.reset()

input_shape = (3, 3, 2)
num_actions = 9

agents = []

for agent_name in env.agents:
    q_network = DQN.get_q_network(input_shape=input_shape, num_actions=num_actions)
    target_q_network = DQN.clone(q_network)
    transition_table = TransitionTable()
    behavior_policy = EpsilonGreedyPolicy(q_network=q_network, epsilon_end=0.2)
    target_policy = GreedyPolicy(q_network=target_q_network)
    agent = AdversarialQLearningAgent(q_network=q_network, target_q_network=target_q_network,
                                      transition_table=transition_table,
                                      behavior_policy=behavior_policy, target_policy=target_policy,
                                      agent_name=agent_name, trainable=True)

    agents.append(agent)

env.reset()

agent_1 = agents[0]
agent_2 = agents[1]

adversarial_q_learning = AdversarialQLearning(env=env, num_actions=num_actions, agent_1=agent_1, agent_2=agent_2)
adversarial_q_learning.train_adversarial(num_iterations=1000)
