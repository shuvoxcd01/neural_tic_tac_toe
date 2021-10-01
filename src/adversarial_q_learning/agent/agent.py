class Agent:
    def __init__(self, q_network, target_q_network, transition_table, behavior_policy,
                 target_policy, agent_name: str, trainable: bool):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.transition_table = transition_table
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.name = agent_name
        self.trainable = trainable
