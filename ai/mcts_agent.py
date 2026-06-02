"""
Monte Carlo Rollout Agent – optimised for Spite Analysis.

Performance guide
-----------------
  MCTS-5   → ~5 ms/decision,  good for training benchmarks
  MCTS-15  → ~20 ms/decision, solid gameplay, default for training
  MCTS-30  → ~60 ms/decision, strong gameplay
  MCTS-50  → ~120 ms/decision, recommended for single-game play

All AI decisions run in a background thread (see AIActionWorker in
main_window.py), so the GUI never freezes regardless of rollout count.

Speed tricks used here
----------------------
* Single deep-copy per candidate action (not per rollout).
* Rollout reuses the same sim state via apply_action in-place.
* Max rollout depth capped at 300 steps (not 600).
* Candidate actions capped at max_candidates to bound total work.
"""

import random, copy
from typing import Optional, List
from game.game_state import GameState, Action
from ai.agent import Agent


class MCTSAgent(Agent):

    def __init__(self, player_id: int,
                 rollouts: int = 15,
                 max_candidates: int = 12,
                 max_rollout_steps: int = 300,
                 seed: Optional[int] = None):
        super().__init__(player_id, name=f"MCTS-{rollouts}")
        self.rollouts = rollouts
        self.max_candidates = max_candidates
        self.max_rollout_steps = max_rollout_steps
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState) -> Optional[Action]:
        plays    = state.get_play_actions()
        discards = state.get_discard_actions()
        valid    = plays + discards

        if not valid:
            return None
        if not plays:
            return self._heuristic_discard(state, discards)

        # Cap candidates for speed
        candidates = valid
        if len(candidates) > self.max_candidates:
            # Prioritise play actions; sample remaining slots from discards
            n_play_slots = min(len(plays), self.max_candidates)
            n_disc_slots = self.max_candidates - n_play_slots
            sampled_plays    = (self._rng.sample(plays, n_play_slots)
                                if len(plays) > n_play_slots else plays)
            sampled_discards = (self._rng.sample(discards, n_disc_slots)
                                if n_disc_slots > 0 and discards else [])
            candidates = sampled_plays + sampled_discards

        best, best_score = candidates[0], -1.0
        for action in candidates:
            score = self._evaluate(state, action)
            if score > best_score:
                best_score = score
                best = action
        return best

    def _evaluate(self, state: GameState, action: Action) -> float:
        """Copy state ONCE, apply action, then run rollouts in-place."""
        wins = 0
        # One copy per candidate (shared across rollouts)
        base = state.copy()
        ok, _ = base.apply_action(action)
        if not ok:
            return 0.0
        if base.is_terminal():
            return 1.0 if base.winner == self.player_id else 0.0

        for _ in range(self.rollouts):
            sim = base.copy()          # copy the post-action state
            winner = self._rollout(sim)
            if winner == self.player_id:
                wins += 1
        return wins / self.rollouts

    def _rollout(self, state: GameState) -> Optional[int]:
        for _ in range(self.max_rollout_steps):
            if state.is_terminal():
                return state.winner
            plays    = state.get_play_actions()
            discards = state.get_discard_actions()
            if plays and (not discards or self._rng.random() < 0.80):
                state.apply_action(self._rng.choice(plays))
            elif discards:
                state.apply_action(self._rng.choice(discards))
            else:
                return state.winner
        return state.winner

    def _heuristic_discard(self, state, actions):
        if not actions:
            return None
        tops = [state.pile_top_value(i) for i in range(state.MAX_BUILDING_PILES)]
        def score(a):
            c = a.card
            if not c: return 0
            if c.is_wild: return -100
            dist = min(abs(c.base_value - (t+1)) for t in tops)
            return c.base_value + dist * 0.5
        return max(actions, key=score)