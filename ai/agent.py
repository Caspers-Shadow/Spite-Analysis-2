from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, env, player_index: int):
        pass
