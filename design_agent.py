"""
design_agent.py - Tabular Q-learning agent for FPGA design space exploration.
"""

import random
import math
import itertools
from collections import defaultdict

# ── Design space ──────────────────────────────────────────────────────────────
PARAM_SPACE = {
    "bit_w":       [4, 8, 16],
    "vec_len":     [2, 4, 8, 16],
    "pipe_stages": [1, 2, 3],
    "act_type":    [0, 1, 2],
    "accum_extra": [2, 4, 8],
    "use_dsp":     [0, 1],
}

ALL_CONFIGS = list(itertools.product(*PARAM_SPACE.values()))
PARAM_NAMES = list(PARAM_SPACE.keys())
TOTAL_CONFIGS = len(ALL_CONFIGS)  # 648


def config_to_dict(config_tuple) -> dict:
    return dict(zip(PARAM_NAMES, config_tuple))


def dict_to_tuple(config_dict) -> tuple:
    return tuple(config_dict[k] for k in PARAM_NAMES)


class FPGADesignAgent:
    """
    Tabular Q-learning agent that navigates the 648-point FPGA design space.

    State  : current config tuple (6-dimensional, discrete)
    Action : pick any config from the full space (simplified: state == config)
    Q(s)   : expected long-term reward for config s
    """

    def __init__(self, lr: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 0.9, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.97):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: config_tuple -> float
        self.q_table: dict[tuple, float] = defaultdict(float)

        # History: list of (config_dict, metrics, reward)
        self.history: list[dict] = []

        # Set of explored configs
        self.explored: set[tuple] = set()

        # Best design found
        self.best_reward: float = -float("inf")
        self.best_entry: dict   = {}

    # ── Config selection ──────────────────────────────────────────────────────

    def select_config(self) -> tuple[dict, str]:
        """
        Epsilon-greedy selection.
        Returns (config_dict, method) where method is 'explore' or 'exploit'.
        """
        if random.random() < self.epsilon:
            # Explore: random config (prefer unexplored)
            unexplored = [c for c in ALL_CONFIGS if c not in self.explored]
            pool = unexplored if unexplored else ALL_CONFIGS
            cfg_tuple = random.choice(pool)
            method = "explore"
        else:
            # Exploit: pick the config with the highest Q-value
            cfg_tuple = max(ALL_CONFIGS, key=lambda c: self.q_table[c])
            method = "exploit"

        return config_to_dict(cfg_tuple), method

    # ── Q-table update ────────────────────────────────────────────────────────

    def update(self, config_dict: dict, reward: float):
        """
        Simple Q-learning update (single-step, no next-state lookahead needed
        because each FPGA config is an independent design point).

        Q(s) <- (1-lr)*Q(s) + lr*(reward + gamma * max_Q)
        Here we treat the max over all configs as the "next" best option.
        """
        s = dict_to_tuple(config_dict)
        best_next_q = max(self.q_table[c] for c in ALL_CONFIGS)
        td_target = reward + self.gamma * best_next_q
        self.q_table[s] = (1 - self.lr) * self.q_table[s] + self.lr * td_target

        # Track exploration
        self.explored.add(s)

        # Record history
        entry = {"config": config_dict.copy(), "reward": reward}
        self.history.append(entry)

        # Update best
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_entry  = entry.copy()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    # ── Top designs ──────────────────────────────────────────────────────────

    def top_designs(self, n: int = 5) -> list[dict]:
        """Return top-n explored configs sorted by Q-value descending."""
        explored_list = sorted(
            self.explored,
            key=lambda c: self.q_table[c],
            reverse=True
        )
        results = []
        for cfg in explored_list[:n]:
            results.append({
                "config": config_to_dict(cfg),
                "q_value": round(self.q_table[cfg], 4),
            })
        return results

    # ── Improvement analysis ──────────────────────────────────────────────────

    def improvement_analysis(self, top_k: int = 10, bot_k: int = 10) -> dict:
        """
        Compare parameter values that appear in the top-k versus bottom-k
        designs (by reward). Highlights which values correlate with quality.
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

            # Values that appear more in top than bottom
            preferred = []
            for v in PARAM_SPACE[param]:
                diff = top_pct.get(v, 0) - bot_pct.get(v, 0)
                if diff > 0.1:
                    preferred.append((v, round(diff, 2)))
            preferred.sort(key=lambda x: -x[1])

            insights[param] = {
                "preferred_values": preferred,
                "top_distribution": {str(k): round(v, 2) for k, v in top_pct.items()},
                "bot_distribution": {str(k): round(v, 2) for k, v in bot_pct.items()},
            }
        return insights

    # ── Epsilon reset ─────────────────────────────────────────────────────────

    def reset_exploration(self, new_epsilon: float = 0.5):
        """Reset epsilon to encourage fresh exploration."""
        self.epsilon = new_epsilon

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary statistics dict."""
        rewards = [e["reward"] for e in self.history]
        n = len(rewards)
        avg_first = sum(rewards[:10]) / max(1, min(10, n))
        avg_last  = sum(rewards[max(0, n - 10):]) / max(1, min(10, n))
        trend = "up" if avg_last > avg_first else "down"
        trend_delta = round(avg_last - avg_first, 4)

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
            "reward_trend_delta":  trend_delta,
        }

    # ── History helpers ───────────────────────────────────────────────────────

    def last_n(self, n: int = 5) -> list[dict]:
        """Return the last n history entries."""
        return self.history[-n:]

    def reward_trend_arrow(self) -> str:
        s = self.stats()
        return "↑" if s["reward_trend"] == "up" else "↓"
