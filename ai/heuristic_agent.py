"""
Heuristic agent for Spite Analysis.

Priority order (highest first):
  1. Play stockpile top onto a building pile  ← directly advances win condition
  2. Play from hand / discard onto building pile that lets stockpile top play next
  3. Play any card from hand onto a building pile
  4. Play any card from discard pile onto a building pile
  5. Discard – choose the card that is least useful
     a. Prefer discarding to a pile that creates a useful ordered sequence
     b. Otherwise discard the highest-value card (frees useful lower numbers)
"""

import random
from typing import List, Optional, Tuple
from game.game_state import GameState, Action
from game.card import Card
from ai.agent import Agent


class HeuristicAgent(Agent):
    """Rule-based heuristic agent."""

    def __init__(self, player_id: int, seed: Optional[int] = None):
        super().__init__(player_id, name="Heuristic")
        self._rng = random.Random(seed)

    # ── Public interface ───────────────────────────────────────────────────────
    def choose_action(self, state: GameState) -> Optional[Action]:
        play_actions = state.get_play_actions()

        if play_actions:
            return self._best_play(state, play_actions)

        # No plays → must discard
        discard_actions = state.get_discard_actions()
        if discard_actions:
            return self._best_discard(state, discard_actions)

        return None

    # ── Play scoring ───────────────────────────────────────────────────────────
    def _best_play(self, state: GameState, actions: List[Action]) -> Action:
        scored = [(self._score_play(state, a), self._rng.random(), a) for a in actions]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]

    def _score_play(self, state: GameState, action: Action) -> float:
        score = 0.0
        player = state.current_player

        # Highest priority: playing from stockpile
        if action.action_type == 'play_stockpile':
            score += 100.0

        # Second: playing a card that will unlock the stockpile top next turn
        elif action.action_type in ('play_hand', 'play_discard'):
            if player.stockpile_top:
                needed_after = state.pile_needed_value(action.target_pile) + 1
                st_val = player.stockpile_top.effective_value() if player.stockpile_top.is_wild \
                    else player.stockpile_top.base_value
                if st_val == needed_after:
                    score += 50.0
                elif abs(st_val - needed_after) <= 2:
                    score += 20.0

        # Bonus: prefer playing lower-value cards (saves higher for later)
        if action.card and not action.card.is_wild:
            score += (12 - action.card.base_value) * 0.5

        # Slight penalty for using wild cards (conserve them)
        if action.card and action.card.is_wild:
            score -= 15.0
            # But if pile is stuck without wild, use it
            needed = state.pile_needed_value(action.target_pile)
            if not self._non_wild_can_fill(state, needed):
                score += 25.0

        # Bonus for extending towards a pile completion
        top_after = state.pile_top_value(action.target_pile) + 1
        score += top_after * 0.3

        return score

    def _non_wild_can_fill(self, state: GameState, value: int) -> bool:
        """Check if any non-wild card in hand/discard/stockpile can fill *value*."""
        player = state.current_player
        sources: List[Optional[Card]] = list(player.hand) + list(player.discard_tops())
        if player.stockpile_top:
            sources.append(player.stockpile_top)
        for c in sources:
            if c and not c.is_wild and c.base_value == value:
                return True
        return False

    # ── Discard scoring ────────────────────────────────────────────────────────
    def _best_discard(self, state: GameState, actions: List[Action]) -> Action:
        scored = [(self._score_discard(state, a), self._rng.random(), a) for a in actions]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]

    def _score_discard(self, state: GameState, action: Action) -> float:
        """Higher score = better discard choice."""
        score = 0.0
        card = action.card
        pile_idx = action.target_pile
        player = state.current_player

        if card is None:
            return 0.0

        # Never discard wild cards if avoidable
        if card.is_wild:
            score -= 50.0

        # Prefer discarding cards that are far from current building pile needs
        build_tops = [state.pile_top_value(i) for i in range(state.MAX_BUILDING_PILES)]
        min_dist = min(abs(card.base_value - (t + 1)) for t in build_tops) if not card.is_wild else 0
        score += min_dist * 2.0

        # Prefer not blocking a useful discard pile (try to keep piles ordered)
        pile_top = player.discard_piles[pile_idx][-1] if player.discard_piles[pile_idx] else None
        if pile_top is not None:
            # Good: place lower value on top of higher value (creates ascending access order)
            if not card.is_wild and not pile_top.is_wild:
                if card.base_value < pile_top.base_value:
                    score += 5.0
                else:
                    score -= 2.0

        # Don't discard a card that matches a current building pile need
        for top in build_tops:
            if card.can_represent(top + 1):
                score -= 10.0
                break

        return score
