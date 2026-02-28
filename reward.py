"""
reward.py - Reward computation for the AI-FPGA Closed Loop.

PRIMARY SIGNAL: inference_accuracy
  The FPGA hardware runs a real neural network. The fraction of FP32
  accuracy retained on the quantized hardware IS the closed-loop signal.

  hardware params → quantization precision → inference accuracy → reward
       ↑                                                              |
       └──────────────── Q-learning ←──────────────────────────────┘

SECONDARY SIGNALS: throughput, resource efficiency, frequency, latency
  These measure how efficiently the hardware delivers that inference.
"""

# ── Normalization targets ──────────────────────────────────────────────────────
NORM = {
    "inference_acc": 1.0,     # fraction of FP32 performance retained  [0,1]
    "throughput":    2000e6,  # 2 GOPS
    "luts":          500,     # LUT budget
    "frequency":     200.0,   # MHz
    "latency":       3,       # clock cycles
}

# ── Reward weights  (must sum to 1.0) ─────────────────────────────────────────
WEIGHTS = {
    "inference_acc": 0.40,   # PRIMARY — does the NN actually work on this HW?
    "throughput":    0.25,   # how fast is inference
    "resource":      0.20,   # silicon area (LUT count)
    "frequency":     0.10,   # timing closure
    "latency":       0.05,   # pipeline depth
}


def compute_reward(metrics: dict, weights: dict = None) -> float:
    """
    Compute a scalar reward in [-1, 1] from a metrics dict.

    Hard gates (checked before weighted scoring):
      synthesis_ok == False      -> -1.0  (invalid RTL)
      hw_pass_rate < 0.95        -> -0.8  (dot-product unit is functionally broken)

    Parameters
    ----------
    metrics : dict
        Required keys:
          synthesis_ok, hw_pass_rate,
          inference_accuracy, fp32_accuracy,
          throughput, luts, frequency, latency
    weights : dict, optional
        Override default WEIGHTS.

    Returns
    -------
    float  reward in [-1.0, 1.0]
    """
    w = weights if weights is not None else WEIGHTS

    # Hard gate 1: synthesis failed
    if not metrics.get("synthesis_ok", True):
        return -1.0

    # Hard gate 2: hardware primitive is broken
    if metrics.get("hw_pass_rate", 1.0) < 0.95:
        return -0.8

    # Inference accuracy (closed-loop primary signal)
    # Score = fpga_accuracy / fp32_accuracy  (fraction of FP32 perf retained)
    fpga_acc = metrics.get("inference_accuracy", 0.0)
    fp32_acc = metrics.get("fp32_accuracy",      1.0)
    s_inf    = max(0.0, min(fpga_acc / max(fp32_acc, 1e-6), 1.0))

    # Throughput: inference OPS/s
    tput         = metrics.get("throughput", 0.0)
    s_throughput = min(tput / NORM["throughput"], 1.0)

    # Resource efficiency: fewer LUTs is better
    luts       = metrics.get("luts", NORM["luts"] * 2)
    s_resource = max(0.0, min(1.0, 1.0 - luts / (2.0 * NORM["luts"])))

    # Clock frequency
    freq        = metrics.get("frequency", 0.0)
    s_frequency = min(freq / NORM["frequency"], 1.0)

    # Pipeline latency
    lat       = metrics.get("latency", NORM["latency"] * 2)
    s_latency = max(0.0, min(1.0, 1.0 - lat / (2.0 * NORM["latency"])))

    reward = (
        w.get("inference_acc", 0.40) * s_inf
        + w.get("throughput",  0.25) * s_throughput
        + w.get("resource",    0.20) * s_resource
        + w.get("frequency",   0.10) * s_frequency
        + w.get("latency",     0.05) * s_latency
    )

    return float(reward)


def breakdown(metrics: dict, weights: dict = None) -> dict:
    """Return per-component scores for debugging / display."""
    if not metrics.get("synthesis_ok", True):
        return {"gate": "synthesis_failed", "reward": -1.0}
    if metrics.get("hw_pass_rate", 1.0) < 0.95:
        return {"gate": "hw_functional_bug", "reward": -0.8}

    fpga_acc = metrics.get("inference_accuracy", 0.0)
    fp32_acc = metrics.get("fp32_accuracy", 1.0)
    s_inf    = min(fpga_acc / max(fp32_acc, 1e-6), 1.0)
    s_tput   = min(metrics.get("throughput", 0) / NORM["throughput"], 1.0)
    s_res    = max(0., min(1., 1. - metrics.get("luts", 0) / (2*NORM["luts"])))
    s_freq   = min(metrics.get("frequency", 0) / NORM["frequency"], 1.0)
    s_lat    = max(0., min(1., 1. - metrics.get("latency", 0) / (2*NORM["latency"])))

    return {
        "inference_score":  round(s_inf,  4),
        "throughput_score": round(s_tput, 4),
        "resource_score":   round(s_res,  4),
        "frequency_score":  round(s_freq, 4),
        "latency_score":    round(s_lat,  4),
        "reward":           round(compute_reward(metrics, weights), 4),
    }
