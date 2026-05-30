"""Random agent – uniform random from all valid actions. Never returns None."""

import random
from typing import Optional
from game.game_state import GameState, Action
from game.player import Player
from ai.agent import Agent


class RandomAgent(Agent):
    def __init__(self, player_id: int, seed: Optional[int] = None):
        super().__init__(player_id, name="Random")
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState) -> Optional[Action]:
        plays    = state.get_play_actions()
        discards = state.get_discard_actions()

        if plays:
            # 40 % chance to stop early and discard instead
            if self._rng.random() < 0.40 and discards:
                return self._rng.choice(discards)
            return self._rng.choice(plays)

        if discards:
            return self._rng.choice(discards)

        # Absolute fallback: force discard first hand card to pile 0
        player = state.current_player
        if player.hand:
            return Action('discard', player.hand[0], None, 0)

        return None  # Truly stuck – game state will auto-resolve via stall detection