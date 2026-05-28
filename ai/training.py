"""
Training infrastructure for Spite Analysis agents.

Supports:
  - AI vs AI self-play (any agent combination)
  - RL agent training with reward shaping
  - Statistics collection and export
  - Model checkpointing
"""

import json
import os
import time
from typing import List, Optional, Callable, Dict, Any, Tuple

from game.game_state import GameState, Action
from ai.agent import Agent
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Single game runner
# ─────────────────────────────────────────────────────────────────────────────

def run_game(
    agent0: Agent,
    agent1: Agent,
    seed: Optional[int] = None,
    max_turns: int = 2000,
    render_callback: Optional[Callable[[GameState, Action], None]] = None,
) -> Dict[str, Any]:
    """
    Run a single complete game between *agent0* (player 0) and *agent1* (player 1).

    Returns a stats dict with winner, turns, duration.
    """
    state = GameState(seed=seed, player_names=(agent0.name, agent1.name))
    agents = {0: agent0, 1: agent1}

    agent0.on_game_start(state)
    agent1.on_game_start(state)

    t0 = time.time()
    stockpile_before = [p.stockpile_size for p in state.players]

    for _ in range(max_turns):
        if state.is_terminal():
            break

        pid = state.current_player_idx
        agent = agents[pid]

        action = agent.choose_action(state)
        if action is None:
            # No action available – skip turn (shouldn't happen in valid game)
            log.warning(f"Agent {agent.name} returned None action on turn {state.turn_number}")
            break

        prev_stocks = [p.stockpile_size for p in state.players]
        success, winner = state.apply_action(action)

        if render_callback:
            render_callback(state, action)

        # Shaped reward for RL agents
        if hasattr(agent, 'record_transition'):
            reward = _shaped_reward(state, pid, prev_stocks)
            agent.record_transition(state, reward, state.is_terminal())

    duration = time.time() - t0
    winner_id = state.winner

    agent0.on_game_end(state, winner_id)
    agent1.on_game_end(state, winner_id)

    return {
        'winner': winner_id,
        'turns': state.turn_number,
        'duration_s': round(duration, 4),
        'completed_sequences': state.completed_sequences,
        'total_plays': state.stats.get('total_plays', 0),
        'total_discards': state.stats.get('total_discards', 0),
        'stockpile_remaining': [p.stockpile_size for p in state.players],
    }


def _shaped_reward(state: GameState, player_id: int, prev_stocks: List[int]) -> float:
    """Dense reward shaping based on stockpile reduction."""
    if state.is_terminal():
        return 1.0 if state.winner == player_id else -1.0

    prev = prev_stocks[player_id]
    curr = state.players[player_id].stockpile_size
    delta = prev - curr   # > 0 if stockpile shrank

    reward = delta * 0.2   # +0.2 per stockpile card played

    # Bonus for opponent stockpile NOT shrinking (relative gain)
    opp_id = 1 - player_id
    opp_delta = prev_stocks[opp_id] - state.players[opp_id].stockpile_size
    if opp_delta == 0 and delta > 0:
        reward += 0.05

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# Training session
# ─────────────────────────────────────────────────────────────────────────────

class TrainingSession:
    """
    Run multiple games for RL training and collect statistics.

    Usage
    -----
    session = TrainingSession(agent0, agent1, n_episodes=5000)
    session.run(callback=my_progress_fn)
    session.save_stats('stats.json')
    """

    def __init__(
        self,
        agent0: Agent,
        agent1: Agent,
        n_episodes: int = 1000,
        checkpoint_dir: str = 'checkpoints',
        checkpoint_freq: int = 500,
        seed: Optional[int] = None,
    ):
        self.agent0 = agent0
        self.agent1 = agent1
        self.n_episodes = n_episodes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.base_seed = seed

        self.results: List[Dict[str, Any]] = []
        self.wins = [0, 0]
        self.episode = 0
        self.running = False
        self._stop_flag = False

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Control ────────────────────────────────────────────────────────────────
    def run(self, callback: Optional[Callable[['TrainingSession'], None]] = None):
        """Run training. *callback* is called after each episode."""
        self.running = True
        self._stop_flag = False

        for ep in range(self.n_episodes):
            if self._stop_flag:
                break

            seed = (self.base_seed + ep) if self.base_seed is not None else None
            result = run_game(self.agent0, self.agent1, seed=seed)
            self.results.append(result)
            self.episode = ep + 1

            if result['winner'] == 0:
                self.wins[0] += 1
            elif result['winner'] == 1:
                self.wins[1] += 1

            if callback:
                callback(self)

            # Checkpoint
            if self.episode % self.checkpoint_freq == 0:
                self._save_checkpoints()

        self._save_checkpoints()
        self.running = False

    def stop(self):
        self._stop_flag = True

    # ── Stats ──────────────────────────────────────────────────────────────────
    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        wins_pct = [w / len(self.results) * 100 for w in self.wins]
        recent = self.results[-50:]
        recent_wins = [sum(1 for r in recent if r['winner'] == i) for i in range(2)]
        avg_turns = sum(r['turns'] for r in self.results) / len(self.results)

        summary = {
            'episodes': self.episode,
            'wins': self.wins,
            'win_pct': [round(w, 1) for w in wins_pct],
            'recent_50_wins': recent_wins,
            'avg_turns': round(avg_turns, 1),
            'avg_duration': round(sum(r['duration_s'] for r in self.results) / len(self.results), 4),
        }

        # RL agent metrics
        for i, agent in enumerate([self.agent0, self.agent1]):
            if hasattr(agent, 'get_stats'):
                summary[f'agent{i}_rl_stats'] = agent.get_stats()

        return summary

    def get_win_rate_history(self, window: int = 50) -> Tuple[List[int], List[float]]:
        """Rolling win rate for player 0 over episodes."""
        episodes_x = []
        win_rates = []
        for i in range(window, len(self.results) + 1, max(1, window // 5)):
            chunk = self.results[max(0, i - window):i]
            rate = sum(1 for r in chunk if r['winner'] == 0) / len(chunk)
            episodes_x.append(i)
            win_rates.append(rate)
        return episodes_x, win_rates

    def save_stats(self, path: str = 'training_stats.json'):
        """Export full results and summary to JSON."""
        data = {
            'summary': self.get_summary(),
            'results': self.results[-500:],   # last 500 to keep file small
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        log.info(f"Stats saved to {path}")

    def _save_checkpoints(self):
        for i, agent in enumerate([self.agent0, self.agent1]):
            if hasattr(agent, 'save'):
                path = os.path.join(self.checkpoint_dir, f'agent{i}_ep{self.episode}.pt')
                agent.save(path)
                log.info(f"Checkpoint saved: {path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    def plot_stats(self, save_path: Optional[str] = None):
        """Plot training statistics using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not available; skipping plot")
            return

        if not self.results:
            return

        episodes_x, win_rates = self.get_win_rate_history(window=50)
        turns = [r['turns'] for r in self.results]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Spite Analysis Training Statistics', fontsize=14, fontweight='bold')

        # Win rate over time
        axes[0, 0].plot(episodes_x, win_rates, color='royalblue', linewidth=2)
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'{self.agent0.name} Win Rate (rolling 50)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)

        # Game length
        window = 20
        smoothed_turns = [
            sum(turns[max(0, i-window):i+1]) / min(i+1, window)
            for i in range(len(turns))
        ]
        axes[0, 1].plot(range(len(smoothed_turns)), smoothed_turns, color='green', linewidth=1.5)
        axes[0, 1].set_title('Average Game Length (turns)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Turns')
        axes[0, 1].grid(True, alpha=0.3)

        # RL loss if available
        if hasattr(self.agent0, 'losses') and self.agent0.losses:
            losses = self.agent0.losses
            axes[1, 0].plot(range(len(losses)), losses, color='orange', alpha=0.7, linewidth=1)
            axes[1, 0].set_title(f'{self.agent0.name} Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No RL loss data', ha='center', va='center',
                            transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Loss')

        # Episode rewards if available
        if hasattr(self.agent0, 'episode_rewards') and self.agent0.episode_rewards:
            rewards = self.agent0.episode_rewards
            axes[1, 1].plot(range(len(rewards)), rewards, color='purple', alpha=0.7, linewidth=1)
            axes[1, 1].set_title(f'{self.agent0.name} Episode Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Total Reward')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Win bar chart instead
            labels = [self.agent0.name, self.agent1.name, 'Draw']
            vals = [self.wins[0], self.wins[1],
                    len(self.results) - self.wins[0] - self.wins[1]]
            colors = ['royalblue', 'salmon', 'grey']
            axes[1, 1].bar(labels, vals, color=colors)
            axes[1, 1].set_title('Total Wins')
            axes[1, 1].set_ylabel('Games Won')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            log.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)
