import random
from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def act(self, env, player_index: int):
        acts = env.get_valid_actions(player_index)
        # filter out END_TURN as random may choose discard or plays
        if not acts:
            return None
        return self.rng.choice(acts)
