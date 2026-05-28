"""
Deep Q-Network (DQN) agent for Spite Analysis.

Architecture
------------
- Observation → flat numeric vector
- Fully-connected network: obs_dim → 256 → 256 → action_dim
- Epsilon-greedy exploration
- Experience replay buffer
- Target network with periodic sync

Dependencies: PyTorch (optional – agent gracefully degrades to random if absent).
"""

import math
import random
import json
import os
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

from game.game_state import GameState, Action
from ai.agent import Agent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Observation / action encoding
# ─────────────────────────────────────────────────────────────────────────────

OBS_DIM = (
    5   # hand values (padded)
    + 5   # hand wild flags
    + 1   # stockpile top value
    + 1   # stockpile top wild
    + 1   # stockpile size (normalised)
    + 4   # discard tops
    + 4   # build tops
    + 1   # opp stockpile top
    + 1   # opp stockpile size
    + 4   # opp discard tops
    + 1   # opp hand size
    + 1   # deck remaining (normalised)
    + 1   # phase
)   # = 30

# Maximum number of discrete actions per step
# plays: 5 hand × 4 piles + 4 discard × 4 piles + 1 stockpile × 4 piles = 36 plays
# discards: 5 hand × 4 discard piles = 20
ACTION_DIM = 56


def encode_observation(state: GameState) -> List[float]:
    """Convert a game state into a fixed-length float vector."""
    obs = state.get_observation()

    def norm_val(v, mx=13.0):
        return v / mx

    vec: List[float] = []

    # Hand (padded to 5)
    hand = obs['hand'][:5] + [0] * max(0, 5 - len(obs['hand']))
    hand_w = obs['hand_wild'][:5] + [0] * max(0, 5 - len(obs['hand_wild']))
    vec += [norm_val(v) for v in hand]
    vec += hand_w

    vec.append(norm_val(obs['stockpile_top']))
    vec.append(float(obs['stockpile_top_wild']))
    vec.append(obs['stockpile_size'] / 13.0)

    dtops = obs['discard_tops'][:4] + [0] * max(0, 4 - len(obs['discard_tops']))
    vec += [norm_val(v) for v in dtops]

    btops = obs['build_tops'][:4] + [0] * max(0, 4 - len(obs['build_tops']))
    vec += [norm_val(v) for v in btops]

    vec.append(norm_val(obs['opp_stockpile_top']))
    vec.append(obs['opp_stockpile_size'] / 13.0)

    odtops = obs['opp_discard_tops'][:4] + [0] * max(0, 4 - len(obs['opp_discard_tops']))
    vec += [norm_val(v) for v in odtops]

    vec.append(obs['opp_hand_size'] / 5.0)
    vec.append(min(obs['deck_remaining'], 104) / 104.0)
    vec.append(float(obs['phase']))

    assert len(vec) == OBS_DIM, f"Expected {OBS_DIM}, got {len(vec)}"
    return vec


def encode_actions(actions: List[Action]) -> List[int]:
    """
    Assign a unique integer ID to each possible action.
    Returns a list of IDs corresponding to the given action list.

    Encoding scheme:
      play_hand: card_slot (0-4) * 4 + pile (0-3)   → 0..19
      play_stockpile: 20 + pile (0-3)                 → 20..23
      play_discard: 24 + src_pile (0-3) * 4 + pile    → 24..39
      discard: 40 + card_slot (0-4) * 4 + discard     → 40..59
    """
    ids = []
    for a in actions:
        if a.action_type == 'play_hand':
            ids.append(a.target_pile)       # simplified: just 0-3
        elif a.action_type == 'play_stockpile':
            ids.append(20 + a.target_pile)
        elif a.action_type == 'play_discard':
            src = a.source_pile or 0
            ids.append(24 + src * 4 + a.target_pile)
        elif a.action_type == 'discard':
            ids.append(40 + a.target_pile)
        else:
            ids.append(0)
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM,
                     hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

        def forward(self, x):
            return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action_id, reward, next_obs, done):
        self.buffer.append((obs, action_id, reward, next_obs, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class RLAgent(Agent):
    """
    DQN-based reinforcement learning agent.
    Falls back to random action selection if PyTorch is not available.
    """

    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 500   # steps
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 10_000         # steps until EPS_END

    def __init__(self, player_id: int, seed: Optional[int] = None,
                 model_path: Optional[str] = None):
        super().__init__(player_id, name="RLAgent")
        self._rng = random.Random(seed)
        self.epsilon = self.EPS_START
        self.steps = 0
        self.training = True

        # Metrics
        self.losses: List[float] = []
        self.episode_rewards: List[float] = []
        self._current_episode_reward = 0.0

        if not _TORCH_AVAILABLE:
            self._net = None
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._net = DQNNetwork().to(self.device)
        self._target_net = DQNNetwork().to(self.device)
        self._target_net.load_state_dict(self._net.state_dict())
        self._target_net.eval()
        self._optimizer = optim.Adam(self._net.parameters(), lr=self.LR)
        self._replay = ReplayBuffer()

        self._last_obs: Optional[List[float]] = None
        self._last_action_id: Optional[int] = 0

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # ── Action selection ───────────────────────────────────────────────────────
    def choose_action(self, state: GameState) -> Optional[Action]:
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return None

        obs = encode_observation(state)

        if not _TORCH_AVAILABLE or self._net is None:
            return self._rng.choice(valid_actions)

        # Epsilon-greedy
        self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-self.steps / self.EPS_DECAY)
        self.steps += 1

        if self._rng.random() < self.epsilon:
            chosen = self._rng.choice(valid_actions)
        else:
            chosen = self._greedy_action(obs, valid_actions)

        # Store for training
        if self.training:
            action_ids = encode_actions(valid_actions)
            chosen_idx = valid_actions.index(chosen) if chosen in valid_actions else 0
            self._last_obs = obs
            self._last_action_id = action_ids[chosen_idx] if chosen_idx < len(action_ids) else 0

        return chosen

    def _greedy_action(self, obs: List[float], valid_actions: List[Action]) -> Action:
        """Return the action with the highest Q-value among valid actions."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self._net(obs_t).squeeze(0).cpu().numpy()

        action_ids = encode_actions(valid_actions)
        best_idx = max(range(len(valid_actions)),
                       key=lambda i: q_vals[action_ids[i] % ACTION_DIM])
        return valid_actions[best_idx]

    # ── Training ───────────────────────────────────────────────────────────────
    def record_transition(self, next_state: GameState, reward: float, done: bool):
        """Store transition in replay buffer and trigger a learning step."""
        if not _TORCH_AVAILABLE or not self.training or self._last_obs is None:
            return
        next_obs = encode_observation(next_state)
        self._replay.push(
            self._last_obs, self._last_action_id, reward, next_obs, done
        )
        self._current_episode_reward += reward
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

        if len(self._replay) >= self.BATCH_SIZE:
            loss = self._train_step()
            self.losses.append(loss)

        if self.steps % self.TARGET_UPDATE_FREQ == 0:
            self._target_net.load_state_dict(self._net.state_dict())

    def _train_step(self) -> float:
        batch = self._replay.sample(self.BATCH_SIZE)
        obs_b, aid_b, rew_b, next_b, done_b = zip(*batch)

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=self.device)
        aid_t = torch.tensor(aid_b, dtype=torch.long, device=self.device)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=self.device)
        next_t = torch.tensor(next_b, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=self.device)

        q_current = self._net(obs_t).gather(1, aid_t.clamp(0, ACTION_DIM - 1).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self._target_net(next_t).max(1)[0]
        q_target = rew_t + self.GAMMA * q_next * (1 - done_t)

        loss = F.smooth_l1_loss(q_current, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
        self._optimizer.step()
        return loss.item()

    # ── Save / load ────────────────────────────────────────────────────────────
    def save(self, path: str):
        if not _TORCH_AVAILABLE or self._net is None:
            return
        torch.save({
            'net': self._net.state_dict(),
            'target': self._target_net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'losses': self.losses[-1000:],
            'episode_rewards': self.episode_rewards[-1000:],
        }, path)

    def load(self, path: str):
        if not _TORCH_AVAILABLE or not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self._net.load_state_dict(ckpt['net'])
        self._target_net.load_state_dict(ckpt['target'])
        self._optimizer.load_state_dict(ckpt['optimizer'])
        self.steps = ckpt.get('steps', 0)
        self.epsilon = ckpt.get('epsilon', self.EPS_END)
        self.losses = ckpt.get('losses', [])
        self.episode_rewards = ckpt.get('episode_rewards', [])

    # ── Stats ──────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        recent_losses = self.losses[-100:] if self.losses else []
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
        return {
            'epsilon': round(self.epsilon, 4),
            'steps': self.steps,
            'avg_loss': round(sum(recent_losses) / len(recent_losses), 6) if recent_losses else 0.0,
            'avg_reward': round(sum(recent_rewards) / len(recent_rewards), 4) if recent_rewards else 0.0,
            'replay_size': len(self._replay) if hasattr(self, '_replay') else 0,
            'torch_available': _TORCH_AVAILABLE,
        }
