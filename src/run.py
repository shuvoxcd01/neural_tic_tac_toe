import gym
import tictactoe

import os
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

# env = gym.make('tic_tac_toe:tic_tac_toe-v0')
from src.q_learning.q_learning import QLearning

env = gym.make('CartPole-v0')

input_shape = env.reset().shape
num_actions = env.action_space.n

q_learning = QLearning(env=env, input_shape=input_shape, num_actions=num_actions)

q_learning.train(3000)
