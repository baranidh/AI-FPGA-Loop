"""
design_agent.py - Deep Q-Network (DQN) agent for FPGA design space exploration.

Upgrades the tabular Q-table to a learned Q-function:

  Architecture (all NumPy, no external ML framework):
    Online network  :  6 → 64 → 32 → 1   (updated every step via Adam)
    Target network  :  same, hard-copied every `target_update` steps
    Experience replay buffer: circular, capacity 2000
    Action selection: ε-greedy — evaluate all 648 configs in one batch forward
                      pass, argmax for exploit; prefer-unexplored random for explore

  Advantages over tabular Q-learning:
    - Generalises across similar configs (nearby bit_w / vec_len share gradients)
    - Scales to much larger design spaces (thousands of configs, continuous params)
    - Training signal from every batch step, not just the single visited config
    - Target network decouples prediction from target → stable convergence

  External interface is identical to the original FPGADesignAgent so
  main_loop.py requires zero changes.
"""

import random
import math
import itertools
import numpy as np
from collections import defaultdict, deque

# ── Design space ───────────────────────────────────────────────────────────────
PARAM_SPACE = {
    "bit_w":       [4, 8, 16],
    "vec_len":     [2, 4, 8, 16],
    "pipe_stages": [1, 2, 3],
    "act_type":    [0, 1, 2],
    "accum_extra": [2, 4, 8],
    "use_dsp":     [0, 1],
}

ALL_CONFIGS   = list(itertools.product(*PARAM_SPACE.values()))
PARAM_NAMES   = list(PARAM_SPACE.keys())
TOTAL_CONFIGS = len(ALL_CONFIGS)   # 648

# Per-parameter [min, max] for normalisation to [0, 1]
_PARAM_LO = {k: min(v) for k, v in PARAM_SPACE.items()}
_PARAM_HI = {k: max(v) for k, v in PARAM_SPACE.items()}


def _encode(cfg_tuple: tuple) -> np.ndarray:
    """Normalise a config tuple to a float32 feature vector in [0, 1]^6."""
    out = []
    for val, name in zip(cfg_tuple, PARAM_NAMES):
        lo, hi = _PARAM_LO[name], _PARAM_HI[name]
        out.append((val - lo) / (hi - lo) if hi != lo else 0.0)
    return np.array(out, dtype=np.float32)


# Pre-compute feature matrix for all 648 configs once at import time
ALL_CONFIG_FEATURES = np.array([_encode(c) for c in ALL_CONFIGS], dtype=np.float32)


def config_to_dict(cfg_tuple: tuple) -> dict:
    return dict(zip(PARAM_NAMES, cfg_tuple))


def dict_to_tuple(cfg_dict: dict) -> tuple:
    return tuple(cfg_dict[k] for k in PARAM_NAMES)


# ── Q-network (3-layer MLP with Adam, pure NumPy) ─────────────────────────────

class _QNetwork:
    """
    3-layer MLP: input(6) → hidden(64, ReLU) → hidden(32, ReLU) → Q-value(1)

    Training: mini-batch MSE with the Adam optimiser.
    He initialisation for all weight matrices.
    """

    def __init__(self, lr: float = 1e-3, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Weight matrices and biases (He init)
        self.W1 = rng.standard_normal((64, 6)).astype(np.float32)  * math.sqrt(2 / 6)
        self.b1 = np.zeros(64,  dtype=np.float32)
        self.W2 = rng.standard_normal((32, 64)).astype(np.float32) * math.sqrt(2 / 64)
        self.b2 = np.zeros(32,  dtype=np.float32)
        self.W3 = rng.standard_normal((1,  32)).astype(np.float32) * math.sqrt(2 / 32)
        self.b3 = np.zeros(1,   dtype=np.float32)

        # Adam hyper-parameters and moment accumulators
        self.lr = lr
        self._beta1, self._beta2, self._eps_adam = 0.9, 0.999, 1e-8
        self._t = 0
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]

        # Cached activations for backprop (populated by forward())
        self._h0 = self._z1 = self._h1 = self._z2 = self._h2 = self._z3 = None

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Caching forward pass used during training. X: (N, 6) → Q: (N,)"""
        self._h0 = X
        self._z1 = X         @ self.W1.T + self.b1   # (N, 64)
        self._h1 = np.maximum(0, self._z1)
        self._z2 = self._h1  @ self.W2.T + self.b2   # (N, 32)
        self._h2 = np.maximum(0, self._z2)
        self._z3 = self._h2  @ self.W3.T + self.b3   # (N,  1)
        return self._z3[:, 0]                          # (N,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Non-caching forward pass for inference. X: (N, 6) → Q: (N,)"""
        h1 = np.maximum(0, X   @ self.W1.T + self.b1)
        h2 = np.maximum(0, h1  @ self.W2.T + self.b2)
        return (h2 @ self.W3.T + self.b3)[:, 0]

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self, X: np.ndarray, targets: np.ndarray):
        """
        One gradient descent step on MSE( Q(X), targets ).
        X: (N, 6)   targets: (N,)
        """
        N   = X.shape[0]
        Q   = self.forward(X)             # (N,) — also caches activations

        # dL/dQ = 2*(Q - targets) / N  (MSE gradient)
        dQ  = (2.0 / N) * (Q - targets)  # (N,)

        # Layer 3 gradients
        dz3 = dQ[:, None]                 # (N, 1)
        dW3 = dz3.T @ self._h2            # (1, 32)
        db3 = dz3.sum(axis=0)             # (1,)
        dh2 = dz3 @ self.W3              # (N, 32)

        # ReLU + layer 2 gradients
        dz2 = dh2 * (self._z2 > 0)       # (N, 32)
        dW2 = dz2.T @ self._h1            # (32, 64)
        db2 = dz2.sum(axis=0)             # (32,)
        dh1 = dz2 @ self.W2              # (N, 64)

        # ReLU + layer 1 gradients
        dz1 = dh1 * (self._z1 > 0)       # (N, 64)
        dW1 = dz1.T @ self._h0            # (64, 6)
        db1 = dz1.sum(axis=0)             # (64,)

        grads  = [dW1, db1, dW2, db2, dW3, db3]
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        # Adam parameter update
        self._t += 1
        t = self._t
        b1c = 1.0 - self._beta1 ** t
        b2c = 1.0 - self._beta2 ** t
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * g
            self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * (g ** 2)
            m_hat = self._m[i] / b1c
            v_hat = self._v[i] / b2c
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps_adam)

    def copy_weights_from(self, src: "_QNetwork"):
        """Hard-copy all weights from src (used for target-network sync)."""
        self.W1[:] = src.W1;  self.b1[:] = src.b1
        self.W2[:] = src.W2;  self.b2[:] = src.b2
        self.W3[:] = src.W3;  self.b3[:] = src.b3


# ── DQN agent (drop-in replacement for the tabular FPGADesignAgent) ───────────

class FPGADesignAgent:
    """
    DQN agent that navigates the 648-point FPGA design space.

    Key differences from the tabular version:
      - Q-values come from a neural network, not a lookup table → generalisation
        across structurally similar configs (e.g. bit_w=8 vs bit_w=16)
      - Experience replay stores (feature_vec, reward) pairs; each gradient
        step trains on a random mini-batch of 32 past experiences
      - A frozen target network computes TD targets, decoupled from the online
        network, which prevents oscillation during early training
      - Action selection (exploit) is a single batch forward pass over all 648
        feature vectors → vectorised, fast, and always up-to-date

    The public interface (select_config, update, top_designs, stats, …) is
    identical to the original tabular FPGADesignAgent.
    """

    def __init__(
        self,
        lr: float             = 0.1,    # CLI --lr; scaled to NN learning rate
        gamma: float          = 0.9,
        epsilon: float        = 0.9,
        epsilon_min: float    = 0.05,
        epsilon_decay: float  = 0.97,
        replay_capacity: int  = 2000,
        batch_size: int       = 32,
        target_update: int    = 20,     # hard target sync every N steps
        nn_lr: float          = 1e-3,   # base NN learning rate (scaled by lr)
    ):
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        # Scale NN lr proportionally to the CLI --lr flag
        effective_lr = nn_lr * (lr / 0.1)
        self.online_net = _QNetwork(lr=effective_lr, seed=42)
        self.target_net = _QNetwork(lr=effective_lr, seed=42)
        self.target_net.copy_weights_from(self.online_net)

        self.replay: deque = deque(maxlen=replay_capacity)

        # History / tracking — same attributes as the tabular agent
        self.history:      list[dict]  = []
        self.explored:     set[tuple]  = set()
        self.best_reward:  float       = -float("inf")
        self.best_entry:   dict        = {}
        self._step_count:  int         = 0

    # ── Config selection ───────────────────────────────────────────────────────

    def select_config(self) -> tuple[dict, str]:
        """
        Epsilon-greedy action selection.

        Exploit: run a single batch forward pass over all 648 config features,
                 return the config with the highest predicted Q-value.
        Explore: uniform random, preferring configs not yet evaluated.
        """
        if random.random() < self.epsilon:
            unexplored = [c for c in ALL_CONFIGS if c not in self.explored]
            pool = unexplored if unexplored else ALL_CONFIGS
            cfg_tuple = random.choice(pool)
            method = "explore"
        else:
            q_vals    = self.online_net.predict(ALL_CONFIG_FEATURES)  # (648,)
            best_idx  = int(np.argmax(q_vals))
            cfg_tuple = ALL_CONFIGS[best_idx]
            method    = "exploit"

        return config_to_dict(cfg_tuple), method

    # ── Network update ────────────────────────────────────────────────────────

    def update(self, config_dict: dict, reward: float):
        """
        1. Store (config_features, reward) in the replay buffer.
        2. If buffer has enough samples, train the online network on a mini-batch.
        3. Every `target_update` steps, hard-copy online → target network.
        4. Decay epsilon.
        """
        cfg_tuple = dict_to_tuple(config_dict)
        feat      = _encode(cfg_tuple)

        self.replay.append((feat, reward))
        self.explored.add(cfg_tuple)

        entry = {"config": config_dict.copy(), "reward": reward}
        self.history.append(entry)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_entry  = entry.copy()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._step_count += 1
        if len(self.replay) >= self.batch_size:
            self._train_batch()

        if self._step_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.online_net)

    def _train_batch(self):
        """Sample a random mini-batch and perform one gradient step."""
        batch         = random.sample(self.replay, self.batch_size)
        feats, rewards = zip(*batch)

        X = np.array(feats,   dtype=np.float32)   # (batch, 6)
        r = np.array(rewards, dtype=np.float32)   # (batch,)

        # TD target: r + γ * max_{a'} Q_target(a')
        target_q_all = self.target_net.predict(ALL_CONFIG_FEATURES)  # (648,)
        max_target_q = float(np.max(target_q_all))
        td_targets   = r + self.gamma * max_target_q                 # (batch,)

        self.online_net.train_step(X, td_targets)

    # ── Top designs ───────────────────────────────────────────────────────────

    def top_designs(self, n: int = 5) -> list[dict]:
        """Return top-n explored configs sorted by network Q-value (descending)."""
        if not self.explored:
            return []
        q_vals = self.online_net.predict(ALL_CONFIG_FEATURES)
        explored_idx = [i for i, c in enumerate(ALL_CONFIGS) if c in self.explored]
        explored_idx.sort(key=lambda i: q_vals[i], reverse=True)
        return [
            {"config": config_to_dict(ALL_CONFIGS[i]), "q_value": round(float(q_vals[i]), 4)}
            for i in explored_idx[:n]
        ]

    # ── Improvement analysis ──────────────────────────────────────────────────

    def improvement_analysis(self, top_k: int = 10, bot_k: int = 10) -> dict:
        """
        Compare parameter distributions in the top-k vs bottom-k results
        (by observed reward). Identical interface to the tabular agent.
        """
        if len(self.history) < 4:
            return {}

        sorted_hist = sorted(self.history, key=lambda e: e["reward"], reverse=True)
        top_entries = sorted_hist[:min(top_k, len(sorted_hist) // 2 + 1)]
        bot_entries = sorted_hist[max(0, len(sorted_hist) - bot_k):]

        insights = {}
        for param in PARAM_NAMES:
            top_counts: dict = defaultdict(int)
            bot_counts: dict = defaultdict(int)
            for e in top_entries:
                top_counts[e["config"][param]] += 1
            for e in bot_entries:
                bot_counts[e["config"][param]] += 1

            top_pct = {v: c / len(top_entries) for v, c in top_counts.items()}
            bot_pct = {v: c / len(bot_entries) for v, c in bot_counts.items()}

            preferred = [
                (v, round(top_pct.get(v, 0) - bot_pct.get(v, 0), 2))
                for v in PARAM_SPACE[param]
                if top_pct.get(v, 0) - bot_pct.get(v, 0) > 0.1
            ]
            preferred.sort(key=lambda x: -x[1])

            insights[param] = {
                "preferred_values": preferred,
                "top_distribution": {str(k): round(v, 2) for k, v in top_pct.items()},
                "bot_distribution": {str(k): round(v, 2) for k, v in bot_pct.items()},
            }
        return insights

    # ── Epsilon reset ─────────────────────────────────────────────────────────

    def reset_exploration(self, new_epsilon: float = 0.5):
        """Bump epsilon to encourage fresh exploration (human review command R)."""
        self.epsilon = new_epsilon

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        rewards   = [e["reward"] for e in self.history]
        n         = len(rewards)
        avg_first = sum(rewards[:10])             / max(1, min(10, n))
        avg_last  = sum(rewards[max(0, n - 10):]) / max(1, min(10, n))
        trend     = "up" if avg_last > avg_first else "down"

        return {
            "iterations":          n,
            "configs_explored":    len(self.explored),
            "total_configs":       TOTAL_CONFIGS,
            "coverage_pct":        round(100 * len(self.explored) / TOTAL_CONFIGS, 1),
            "best_reward":         round(self.best_reward, 4),
            "epsilon":             round(self.epsilon, 4),
            "avg_reward_first10":  round(avg_first, 4),
            "avg_reward_last10":   round(avg_last, 4),
            "reward_trend":        trend,
            "reward_trend_delta":  round(avg_last - avg_first, 4),
            "replay_buffer_size":  len(self.replay),
            "nn_train_steps":      self._step_count,
        }

    # ── History helpers ───────────────────────────────────────────────────────

    def last_n(self, n: int = 5) -> list[dict]:
        return self.history[-n:]

    def reward_trend_arrow(self) -> str:
        return "↑" if self.stats()["reward_trend"] == "up" else "↓"
