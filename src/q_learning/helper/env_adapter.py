from tic_tac_toe.envs.tic_tac_toe_base_env import TicTacToeBaseEnv
import numpy as np


class TicTacToeEnvAdapter:
    def __init__(self, env: TicTacToeBaseEnv):
        self.env = env

    def __getattr__(self, item):
        return getattr(self.env, item)

    def reset(self):
        observation = self.env.reset()
        observation = np.append(observation, float(self.env.player_to_move))
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.append(observation, float(self.env.player_to_move))
        return observation, reward, done, info
