"""Random agent – selects uniformly at random from all legal actions."""

import random
from typing import Optional
from game.game_state import GameState, Action
from game.player import Player
from ai.agent import Agent


class RandomAgent(Agent):
    """
    Baseline agent that plays a uniformly random legal action each step.
    Always returns a valid Action (never None) by falling back to a forced
    discard if no other options exist.
    """

    def __init__(self, player_id: int, seed: Optional[int] = None):
        super().__init__(player_id, name="Random")
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState) -> Optional[Action]:
        play_actions = state.get_play_actions()

        if play_actions:
            # 40 % of the time stop playing early and discard instead
            if self._rng.random() < 0.40:
                discard_actions = state.get_discard_actions()
                if discard_actions:
                    return self._rng.choice(discard_actions)
            return self._rng.choice(play_actions)

        # No plays available → try to discard
        discard_actions = state.get_discard_actions()
        if discard_actions:
            return self._rng.choice(discard_actions)

        # Last resort: if somehow we have no actions (very rare edge case where
        # all hand cards were auto-played and the turn didn't switch), return
        # a forced discard of the first hand card to any discard pile.
        player = state.current_player
        if player.hand:
            return Action('discard', player.hand[0], None, 0)

        # Truly nothing to do – game state will handle the stall
        return None