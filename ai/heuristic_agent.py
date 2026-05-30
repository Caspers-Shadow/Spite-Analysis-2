"""
Heuristic agent for Spite Analysis.

Priority order (highest first)
-------------------------------
WHEN NOT UNLOCKED (player hasn't placed an Ace yet):
  1. Play an Ace from hand / stockpile / discard to unlock immediately
  2. Discard strategically to access Aces faster

WHEN UNLOCKED:
  1. Play stockpile top
  2. Play a card that sets up the stockpile top for next turn
  3. Play any card from hand / discard onto a building pile
  4. Conserve wild cards (use only when no non-wild option exists)
  5. Discard: avoid burying useful low cards, never discard wilds
"""

import random
from typing import List, Optional
from game.game_state import GameState, Action
from game.card import Card
from ai.agent import Agent


class HeuristicAgent(Agent):

    def __init__(self, player_id: int, seed: Optional[int] = None):
        super().__init__(player_id, name="Heuristic")
        self._rng = random.Random(seed)

    # ── Main entry point ───────────────────────────────────────────────────────
    def choose_action(self, state: GameState) -> Optional[Action]:
        play_actions = state.get_play_actions()
        if play_actions:
            return self._best_play(state, play_actions)
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
        pid = state.current_player_idx
        unlocked = state.player_unlocked[pid]

        # Determine the value this action plays
        play_value = action.wild_value if action.wild_value else (
            action.card.base_value if action.card else 0)

        # ── Unlock rule: huge bonus for playing an Ace when not yet unlocked ──
        if not unlocked:
            if play_value == 1:
                score += 200.0   # top priority: get unlocked!
            else:
                score -= 999.0   # shouldn't happen (get_play_actions filters these out)

        # ── Stockpile play: highest regular priority ──────────────────────────
        if action.action_type == 'play_stockpile':
            score += 100.0

        # ── Hand / discard play: reward if it sets up the stockpile ──────────
        elif action.action_type in ('play_hand', 'play_discard'):
            if player.stockpile_top:
                st_val = (player.stockpile_top.effective_value()
                          if player.stockpile_top.is_wild
                          else player.stockpile_top.base_value)
                needed_after = state.pile_needed_value(action.target_pile) + 1
                if st_val == needed_after:
                    score += 50.0
                elif abs(st_val - needed_after) <= 2:
                    score += 15.0

        # ── Wild card penalty: conserve wilds ────────────────────────────────
        if action.card and action.card.is_wild:
            score -= 20.0
            # But if no non-wild can fill this slot, reduce penalty
            needed = state.pile_needed_value(action.target_pile)
            if not self._non_wild_available(state, needed):
                score += 30.0

        # ── Prefer lower play values (leave higher for later) ────────────────
        if play_value and not action.card.is_wild:
            score += (12 - play_value) * 0.4

        # ── Bonus for advancing a pile toward completion ──────────────────────
        score += play_value * 0.2 if play_value else 0

        return score

    def _non_wild_available(self, state: GameState, value: int) -> bool:
        p = state.current_player
        sources = list(p.hand) + list(p.discard_tops())
        if p.stockpile_top: sources.append(p.stockpile_top)
        return any(c and not c.is_wild and c.base_value == value for c in sources if c)

    # ── Discard scoring ────────────────────────────────────────────────────────
    def _best_discard(self, state: GameState, actions: List[Action]) -> Action:
        scored = [(self._score_discard(state, a), self._rng.random(), a) for a in actions]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]

    def _score_discard(self, state: GameState, action: Action) -> float:
        score = 0.0
        card = action.card
        pile_idx = action.target_pile
        player = state.current_player
        pid = state.current_player_idx

        if card is None:
            return 0.0

        # Never discard wilds
        if card.is_wild:
            score -= 80.0

        # When NOT unlocked: prioritise discarding high-value non-Ace cards
        # to cycle through the deck faster and reach Aces
        if not state.player_unlocked[pid]:
            if card.base_value != 1:
                score += card.base_value * 1.5   # prefer discarding highest non-Ace
            else:
                score -= 200.0   # NEVER discard an Ace when not yet unlocked

        # Prefer discarding cards far from current building pile needs
        build_tops = [state.pile_top_value(i) for i in range(state.MAX_BUILDING_PILES)]
        min_dist = min((abs(card.base_value - (t + 1)) for t in build_tops), default=0)
        score += min_dist * 1.5

        # Avoid discarding a card that directly fills a building pile need
        for top in build_tops:
            if card.can_represent(top + 1):
                score -= 12.0
                break

        # Prefer placing on a pile where we can create ordered sequences
        pile = player.discard_piles[pile_idx]
        if pile:
            top_val = pile[-1].base_value if not pile[-1].is_wild else 0
            if not card.is_wild and top_val > card.base_value:
                score += 4.0   # card value < pile top: good stacking order
            else:
                score -= 1.0

        return score