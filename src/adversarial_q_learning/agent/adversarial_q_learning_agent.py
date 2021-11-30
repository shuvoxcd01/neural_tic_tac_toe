import copy
from typing import Optional

from src.adversarial_q_learning.agent.agent import Agent
from src.adversarial_q_learning.network.dqn import DQN
from src.adversarial_q_learning.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from src.adversarial_q_learning.policies.greedy_policy import GreedyPolicy
from src.adversarial_q_learning.transition_table.transition_table import TransitionTable


class AdversarialQLearningAgent(Agent):
    def __init__(self, q_network, target_q_network, transition_table: TransitionTable,
                 behavior_policy: EpsilonGreedyPolicy, target_policy: GreedyPolicy, agent_name: str,
                 trainable: bool = True):
        super().__init__(q_network, target_q_network, transition_table, behavior_policy, target_policy, agent_name,
                         trainable)
        self.behavior_policy.set_q_network(q_network=self.q_network)
        self.target_policy.set_q_network(q_network=self.target_q_network)

    def clone(self, set_trainable: Optional[bool] = None):
        cloned_agent: AdversarialQLearningAgent = copy.deepcopy(self)
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

    def reset_behavior_policy(self):
        self.behavior_policy.reset_epsilon()

    def set_trainable(self, trainable: bool):
        self.trainable = trainable
