import copy
from typing import Optional

from src.q_learning.network.dqn import DQN


class Agent:
    def __init__(self, q_network, target_q_network, transition_table, behavior_policy, target_policy, agent_name,
                 is_trainable=True):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.transition_table = transition_table
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.name = agent_name
        self.trainable = is_trainable

    def clone(self, set_trainable: Optional[bool] = None):
        cloned_agent = copy.deepcopy(self)
        q_network = DQN.clone(self.q_network)
        target_q_network = DQN.clone(self.target_q_network)

        cloned_agent.q_network = q_network
        cloned_agent.target_q_network = target_q_network
        cloned_agent.behavior_policy.set_q_network(q_network=q_network)
        cloned_agent.target_policy.set_q_network(q_network=target_q_network)

        if set_trainable is not None:
            cloned_agent.trainable = set_trainable

        return cloned_agent

    def clear_transition_table(self):
        self.transition_table.clear()
