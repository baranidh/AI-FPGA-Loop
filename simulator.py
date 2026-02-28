"""
simulator.py - Pure Python / NumPy hardware simulator for the dot-product unit.

Provides:
  simulate_dot_product   - bit-exact fixed-point computation matching the Verilog
  run_accuracy_test      - compare fixed-point vs FP32 over many random vectors
  estimate_synthesis_metrics - model LUTs/FFs/frequency from config
  run_full_simulation    - combine all of the above into a metrics dict
"""

import math
import numpy as np
import random


# ── Fixed-point helpers ───────────────────────────────────────────────────────

def _signed_clamp(value: int, bits: int) -> int:
    """Clamp a Python int to a signed two's-complement range."""
    lo = -(1 << (bits - 1))
    hi =  (1 << (bits - 1)) - 1
    return max(lo, min(hi, int(value)))


def simulate_dot_product(
    weights: list[int],
    inputs:  list[int],
    bias:    int,
    config:  dict,
) -> int:
    """
    Bit-exact fixed-point dot product matching the generated Verilog.

    Parameters
    ----------
    weights, inputs : list[int]
        Signed integers in range [-(2^(bit_w-1)), 2^(bit_w-1)-1].
    bias : int
        Signed integer in same range.
    config : dict
        Keys: bit_w, vec_len, pipe_stages, act_type, accum_extra, use_dsp

    Returns
    -------
    int
        Signed integer in bit_w range.
    """
    bit_w       = config["bit_w"]
    vec_len     = config["vec_len"]
    act_type    = config["act_type"]
    accum_extra = config["accum_extra"]

    accum_w = bit_w * 2 + accum_extra + int(math.ceil(math.log2(vec_len)))

    # Accumulate products using int64 to match Verilog sign extension
    accum = np.int64(0)
    for w, x in zip(weights[:vec_len], inputs[:vec_len]):
        prod = np.int64(w) * np.int64(x)
        accum += prod

    accum += np.int64(bias)

    clamp_max = (1 << (bit_w - 1)) - 1
    clamp_min = -(1 << (bit_w - 1))

    # Activation
    if act_type == 0:   # none — truncate
        result = int(accum) & ((1 << bit_w) - 1)
        # reinterpret as signed
        if result >= (1 << (bit_w - 1)):
            result -= (1 << bit_w)
    elif act_type == 1:  # ReLU
        if accum < 0:
            result = 0
        elif accum > clamp_max:
            result = clamp_max
        else:
            result = int(accum)
    else:               # clamp
        result = _signed_clamp(int(accum), bit_w)

    return result


# ── Accuracy test ─────────────────────────────────────────────────────────────

def _fp32_reference(weights, inputs, bias, config) -> float:
    """FP32 reference that applies the same activation as the fixed-point path."""
    bit_w    = config["bit_w"]
    act_type = config["act_type"]
    vec_len  = config["vec_len"]
    clamp_max = float((1 << (bit_w - 1)) - 1)
    clamp_min = float(-(1 << (bit_w - 1)))

    accum = sum(float(w) * float(x) for w, x in zip(weights[:vec_len], inputs[:vec_len]))
    accum += float(bias)

    if act_type == 0:   # truncate — just clamp for reference
        accum = max(clamp_min, min(clamp_max, accum))
    elif act_type == 1:  # ReLU
        accum = max(0.0, min(clamp_max, accum))
    else:               # clamp
        accum = max(clamp_min, min(clamp_max, accum))

    return accum


def run_accuracy_test(config: dict, n_tests: int = 256) -> dict:
    """
    Compare fixed-point dot product vs FP32 reference over random test vectors.
    Inputs are scaled to 1/4 of maximum range to reduce overflow.

    Returns
    -------
    dict with keys: mae, pass_rate, n_tests
    """
    bit_w   = config["bit_w"]
    vec_len = config["vec_len"]

    lo = -(1 << (bit_w - 1))
    hi =  (1 << (bit_w - 1)) - 1
    # Use quarter-range inputs to keep accumulator in-range more often
    scale_lo = lo // 4
    scale_hi = hi // 4

    rng = np.random.default_rng(42)
    errors = []
    passes = 0

    for _ in range(n_tests):
        w    = rng.integers(scale_lo, scale_hi + 1, size=vec_len).tolist()
        x    = rng.integers(scale_lo, scale_hi + 1, size=vec_len).tolist()
        bias = int(rng.integers(scale_lo, scale_hi + 1))

        # FP32 reference (same activation applied)
        fp32_ref = _fp32_reference(w, x, bias, config)

        # Fixed-point result
        fp_out = simulate_dot_product(w, x, bias, config)

        # Normalise by the possible output range
        scale = float(hi - lo) if hi != lo else 1.0
        err = abs(fp_out - fp32_ref) / scale
        errors.append(err)

        # Pass if error < 10% of output range (lenient for quantization noise)
        if err < 0.10:
            passes += 1

    mae = float(np.mean(errors))
    pass_rate = passes / n_tests
    return {"mae": mae, "pass_rate": pass_rate, "n_tests": n_tests}


# ── Synthesis estimation ──────────────────────────────────────────────────────

def estimate_synthesis_metrics(config: dict) -> dict:
    """
    Model LUT / FF / frequency from config parameters.
    Calibrated against typical Yosys results for Xilinx-like FPGA targets.

    Returns
    -------
    dict with keys: luts, ffs, frequency, latency, synthesis_ok
    """
    bit_w       = config["bit_w"]
    vec_len     = config["vec_len"]
    pipe_stages = config["pipe_stages"]
    act_type    = config["act_type"]
    accum_extra = config["accum_extra"]
    use_dsp     = config["use_dsp"]

    # ── LUT estimation ────────────────────────────────────────────────────────
    if use_dsp:
        mult_luts = 3 * vec_len         # DSPs use very few LUTs
    else:
        # LUT-based multiplier: scales super-linearly with bit width
        mult_luts = int((bit_w ** 1.4) * vec_len)

    # Adder tree: log2 levels × accum_w / 4 LUTs per adder
    accum_w   = bit_w * 2 + accum_extra + int(math.ceil(math.log2(max(vec_len, 2))))
    adder_luts = int(math.ceil(math.log2(max(vec_len, 2))) * accum_w / 4)

    # Activation function
    act_luts = {0: 0, 1: bit_w // 2, 2: bit_w}[act_type]

    # Control / misc
    misc_luts = 8

    luts = mult_luts + adder_luts + act_luts + misc_luts

    # ── FF estimation ─────────────────────────────────────────────────────────
    # Each pipeline stage adds bit_w flip-flops
    ffs = pipe_stages * bit_w + accum_w  # accumulator register too

    # ── DSP block count ───────────────────────────────────────────────────────
    dsps = vec_len if use_dsp else 0

    # ── Critical path delay (ns) ──────────────────────────────────────────────
    if use_dsp:
        mult_delay = 2.5
    else:
        mult_delay = 0.08 * (bit_w ** 1.5)

    adder_delay = 0.05 * accum_w
    act_delay   = {0: 0.0, 1: 0.5, 2: 0.8}[act_type]

    # Pipelining divides the critical path
    cp_delay = (mult_delay + adder_delay + act_delay) / pipe_stages

    # Add clock uncertainty / setup-hold margin
    cp_delay += 0.5

    # Frequency in MHz
    frequency = min(500.0, 1000.0 / max(cp_delay, 0.1))

    # Latency in clock cycles = pipe_stages + 1 (output register)
    latency = pipe_stages + 1

    return {
        "luts":        int(luts),
        "ffs":         int(ffs),
        "dsps":        int(dsps),
        "frequency":   round(frequency, 2),
        "latency":     latency,
        "synthesis_ok": True,
    }


# ── Full simulation (sim mode) ────────────────────────────────────────────────

def run_full_simulation(config: dict, quiet: bool = False) -> dict:
    """
    Run the complete sim-mode evaluation pipeline:
      1. Estimate synthesis metrics
      2. Run accuracy test
      3. Combine into a unified metrics dict

    Returns
    -------
    dict with all metrics needed by reward.compute_reward()
    """
    synth  = estimate_synthesis_metrics(config)
    acctest = run_accuracy_test(config)

    freq_hz    = synth["frequency"] * 1e6
    throughput = freq_hz * config["vec_len"]   # ops/sec
    ops_per_lut = throughput / max(synth["luts"], 1)

    metrics = {
        "synthesis_ok": synth["synthesis_ok"],
        "luts":         synth["luts"],
        "ffs":          synth["ffs"],
        "dsps":         synth["dsps"],
        "frequency":    synth["frequency"],
        "latency":      synth["latency"],
        "throughput":   throughput,
        "ops_per_lut":  ops_per_lut,
        "mae":          acctest["mae"],
        "pass_rate":    acctest["pass_rate"],
        "n_tests":      acctest["n_tests"],
        "mode":         "sim",
    }
    return metrics
