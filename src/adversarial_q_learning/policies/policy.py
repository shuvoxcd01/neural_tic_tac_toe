from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def get_action(self, observation, action_mask):
        raise NotImplementedError()
