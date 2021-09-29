import tensorflow as tf
from pettingzoo.classic import tictactoe_v3

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")
from src.q_learning.agent.agent import Agent
from src.q_learning.network.dqn import DQN
from src.q_learning.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from src.q_learning.policies.greedy_policy import GreedyPolicy
from src.q_learning.policies.human_policy import HumanPolicy
from src.q_learning.transition_table.transition_table import TransitionTable

env = tictactoe_v3.env()
env.reset()

input_shape = (3, 3, 2)
num_actions = 9

agents = {}

q_network = DQN.get_q_network(input_shape=input_shape, num_actions=num_actions)
target_q_network = DQN.clone(q_network)
transition_table = TransitionTable()
behavior_policy = EpsilonGreedyPolicy(q_network=q_network)
target_policy = GreedyPolicy(q_network=target_q_network)
dqn_agent = Agent(q_network=q_network, target_q_network=target_q_network, transition_table=transition_table,
                  behavior_policy=behavior_policy, target_policy=target_policy, agent_name="DQN_Agent")

agents[dqn_agent.name] = dqn_agent

human_policy = HumanPolicy(num_actions=num_actions)
human_agent = Agent(q_network=None, target_q_network=None, transition_table=None, behavior_policy=None,
                    target_policy=human_policy, agent_name="Human_Agent")
agents[human_agent.name] = human_agent

env.reset()
env.render()
for _ in env.agent_iter():
    for agent in agents.values():
        observation_with_action_mask, reward, done, _ = env.last(observe=True)

        observation = observation_with_action_mask["observation"]
        action_mask = observation_with_action_mask["action_mask"]

        action = agent.target_policy.get_action(tf.expand_dims(tf.identity(observation), 0),
                                                action_mask).numpy() if not done else None

        env.step(action)
        env.render()
