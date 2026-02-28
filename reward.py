"""
reward.py - Reward computation for AI-FPGA-Loop
Combines performance metrics into a scalar reward for Q-learning.
"""

# Normalization targets (approximate best-case values)
NORM_TARGETS = {
    "throughput": 2000e6,   # 2 GOPS
    "accuracy":   0.001,    # MAE < 0.001 (low is good, so inverted)
    "luts":       500,      # 500 LUTs baseline
    "frequency":  200.0,    # 200 MHz
    "latency":    3,        # 3 clock cycles
}

# Reward weights (must sum to 1.0)
WEIGHTS = {
    "throughput":  0.30,
    "accuracy":    0.20,
    "resource":    0.25,
    "frequency":   0.15,
    "latency":     0.10,
}


def compute_reward(metrics: dict, weights: dict = None) -> float:
    """
    Compute a scalar reward in [-1, 1] from a metrics dict.

    Hard gates:
      - synthesis_ok == False  -> return -1.0
      - pass_rate < 0.95       -> return -0.8

    Parameters
    ----------
    metrics : dict
        Keys: synthesis_ok, pass_rate, throughput, mae, luts, frequency, latency
    weights : dict, optional
        Override default WEIGHTS.

    Returns
    -------
    float
        Scalar reward in [-1.0, 1.0].
    """
    w = weights if weights is not None else WEIGHTS

    # Hard gate 1: synthesis failed
    if not metrics.get("synthesis_ok", True):
        return -1.0

    # Hard gate 2: functional bugs
    if metrics.get("pass_rate", 1.0) < 0.95:
        return -0.8

    # --- throughput score: higher is better ---
    tput = metrics.get("throughput", 0.0)
    s_throughput = min(tput / NORM_TARGETS["throughput"], 1.0)

    # --- accuracy score: lower MAE is better ---
    mae = metrics.get("mae", 1.0)
    # map [0, 2*target] -> [1, 0]
    s_accuracy = max(0.0, 1.0 - mae / (2.0 * NORM_TARGETS["accuracy"]))
    s_accuracy = min(s_accuracy, 1.0)

    # --- resource efficiency: fewer LUTs is better ---
    luts = metrics.get("luts", NORM_TARGETS["luts"] * 2)
    s_resource = max(0.0, 1.0 - luts / (2.0 * NORM_TARGETS["luts"]))
    s_resource = min(s_resource, 1.0)

    # --- frequency score: higher is better ---
    freq = metrics.get("frequency", 0.0)
    s_frequency = min(freq / NORM_TARGETS["frequency"], 1.0)

    # --- latency score: lower is better ---
    lat = metrics.get("latency", NORM_TARGETS["latency"] * 2)
    s_latency = max(0.0, 1.0 - lat / (2.0 * NORM_TARGETS["latency"]))
    s_latency = min(s_latency, 1.0)

    # Weighted sum
    reward = (
        w.get("throughput", 0.30) * s_throughput
        + w.get("accuracy",  0.20) * s_accuracy
        + w.get("resource",  0.25) * s_resource
        + w.get("frequency", 0.15) * s_frequency
        + w.get("latency",   0.10) * s_latency
    )

    return float(reward)


def breakdown(metrics: dict, weights: dict = None) -> dict:
    """Return per-component scores for debugging."""
    w = weights if weights is not None else WEIGHTS
    if not metrics.get("synthesis_ok", True):
        return {"gate": "synthesis_failed", "reward": -1.0}
    if metrics.get("pass_rate", 1.0) < 0.95:
        return {"gate": "functional_bug", "reward": -0.8}

    tput   = metrics.get("throughput", 0.0)
    mae    = metrics.get("mae", 1.0)
    luts   = metrics.get("luts", NORM_TARGETS["luts"] * 2)
    freq   = metrics.get("frequency", 0.0)
    lat    = metrics.get("latency", NORM_TARGETS["latency"] * 2)

    s_throughput = min(tput / NORM_TARGETS["throughput"], 1.0)
    s_accuracy   = max(0.0, min(1.0, 1.0 - mae / (2.0 * NORM_TARGETS["accuracy"])))
    s_resource   = max(0.0, min(1.0, 1.0 - luts / (2.0 * NORM_TARGETS["luts"])))
    s_frequency  = min(freq / NORM_TARGETS["frequency"], 1.0)
    s_latency    = max(0.0, min(1.0, 1.0 - lat / (2.0 * NORM_TARGETS["latency"])))

    return {
        "throughput_score": round(s_throughput, 4),
        "accuracy_score":   round(s_accuracy, 4),
        "resource_score":   round(s_resource, 4),
        "frequency_score":  round(s_frequency, 4),
        "latency_score":    round(s_latency, 4),
        "reward": round(compute_reward(metrics, w), 4),
    }
