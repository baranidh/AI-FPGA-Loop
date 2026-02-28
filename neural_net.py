"""
neural_net.py - TinyMLP that runs on the FPGA dot-product hardware.

THE CLOSED LOOP
---------------
  hardware params → quantization precision → NN inference accuracy → reward
       ↑                                                                  |
       └──────────── Q-learning selects better hardware ←────────────────┘
                          +
  every K iters: QAT fine-tunes model weights for current best hardware
       → model learns to work within hardware constraints
       → better model → higher reward for same hardware → agent keeps it

Architecture : 8 inputs → 16 hidden (ReLU) → 4 output classes
Training     : FP32 mini-batch SGD with cross-entropy
Inference    : fixed-point, using the FPGA's bit_w / act_type / vec_len
"""

import math
import numpy as np


# ── Quantization helpers ───────────────────────────────────────────────────────

def quantize(arr: np.ndarray, bit_w: int,
             scale: float = None) -> tuple[np.ndarray, float]:
    """
    Symmetric per-tensor linear quantization.
    Returns (int64 array in [clamp_min, clamp_max], scale).
    """
    clamp_max = (1 << (bit_w - 1)) - 1
    clamp_min = -(1 << (bit_w - 1))
    if scale is None:
        abs_max = float(np.abs(arr).max())
        scale   = abs_max / clamp_max if abs_max > 0 else 1.0
    q = np.clip(np.round(arr / scale), clamp_min, clamp_max).astype(np.int64)
    return q, scale


# ── FPGA layer forward ─────────────────────────────────────────────────────────

def fpga_layer(X_q: np.ndarray, W_q: np.ndarray, b_q: np.ndarray,
               config: dict, apply_act: bool = True) -> np.ndarray:
    """
    One fully-connected layer using FPGA fixed-point arithmetic.

    Mirrors the hardware exactly:
      - vec_len multiplications happen in parallel per clock cycle
      - bit_w determines the width of operands and the activation clamp
      - act_type is the hardware activation applied after accumulation

    Parameters
    ----------
    X_q  : (n, in_dim)   int64 — quantized activations or inputs
    W_q  : (in_dim, out) int64 — quantized weight matrix (column = one neuron)
    b_q  : (out,)        int64 — quantized bias
    config : FPGA hardware config dict
    apply_act : whether to apply the FPGA's activation function

    Returns
    -------
    np.ndarray (n, out) int64
    """
    bit_w    = config["bit_w"]
    vec_len  = config["vec_len"]
    act_type = config["act_type"]
    clamp_max = np.int64((1 << (bit_w - 1)) - 1)
    clamp_min = np.int64(-(1 << (bit_w - 1)))

    n, in_dim = X_q.shape
    out_dim   = W_q.shape[1]
    result    = np.zeros((n, out_dim), dtype=np.int64)

    for j in range(out_dim):
        w_col = W_q[:, j]          # (in_dim,)
        acc   = np.zeros(n, dtype=np.int64)

        # Accumulate in chunks of vec_len (mirrors the FPGA parallelism)
        for start in range(0, in_dim, vec_len):
            end     = min(start + vec_len, in_dim)
            chunk_w = w_col[start:end]            # (chunk,)  int64
            chunk_x = X_q[:, start:end]           # (n, chunk) int64
            acc    += chunk_x @ chunk_w           # vectorised over samples

        acc += b_q[j]

        if apply_act:
            if act_type == 1:       # ReLU
                acc = np.where(acc < 0, np.int64(0),
                      np.where(acc > clamp_max, clamp_max, acc))
            elif act_type == 2:     # symmetric clamp
                acc = np.clip(acc, clamp_min, clamp_max)
            else:                   # act_type == 0: wrap (truncate)
                acc = ((acc % (1 << bit_w)) + (1 << bit_w)) % (1 << bit_w)
                acc = np.where(acc >= (1 << (bit_w - 1)),
                               acc - (1 << bit_w), acc)

        result[:, j] = acc

    return result


# ── TinyMLP ────────────────────────────────────────────────────────────────────

class TinyMLP:
    """
    Two-layer MLP:  8 → 16 (ReLU) → 4

    Designed to fit on the FPGA dot-product hardware:
      • Layer 1: 8-wide inputs  → each of 16 neurons is one dot-product
      • Layer 2: 16-wide inputs → each of  4 neurons is one dot-product
    The hardware's vec_len, bit_w, and act_type directly govern inference.
    """

    def __init__(self, n_in: int = 8, n_hidden: int = 16, n_out: int = 4,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in     = n_in
        self.n_hidden = n_hidden
        self.n_out    = n_out
        # He initialisation
        self.W1 = rng.standard_normal((n_in,     n_hidden)) * math.sqrt(2 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_out))    * math.sqrt(2 / n_hidden)
        self.b2 = np.zeros(n_out)

    # ── FP32 helpers ──────────────────────────────────────────────────────────

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        return E / E.sum(axis=1, keepdims=True)

    def fp32_forward(self, X: np.ndarray) -> np.ndarray:
        H = np.maximum(0.0, X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2

    def fp32_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((np.argmax(self.fp32_forward(X), axis=1) == y).mean())

    # ── FP32 training ─────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 250, lr: float = 0.02, batch: int = 64,
              verbose: bool = False) -> list[float]:
        """Mini-batch SGD with cross-entropy. Returns per-epoch loss."""
        n   = len(X)
        rng = np.random.default_rng(99)
        losses = []

        for ep in range(epochs):
            idx = rng.permutation(n)
            Xs, ys = X[idx], y[idx]
            ep_loss = 0.0

            for s in range(0, n, batch):
                Xb, yb = Xs[s:s+batch], ys[s:s+batch]
                nb = len(Xb)

                # Forward
                H_pre = Xb @ self.W1 + self.b1
                H     = np.maximum(0.0, H_pre)
                logits = H @ self.W2 + self.b2
                probs  = self._softmax(logits)

                # Loss
                ep_loss += -np.log(probs[np.arange(nb), yb] + 1e-12).mean()

                # Backward
                oh  = np.zeros_like(probs); oh[np.arange(nb), yb] = 1.0
                dZ2 = (probs - oh) / nb
                dW2 = H.T  @ dZ2;   db2 = dZ2.sum(0)
                dH  = dZ2  @ self.W2.T
                dH1 = dH * (H_pre > 0)
                dW1 = Xb.T @ dH1;   db1 = dH1.sum(0)

                self.W1 -= lr * dW1;  self.b1 -= lr * db1
                self.W2 -= lr * dW2;  self.b2 -= lr * db2

            losses.append(ep_loss)
            if verbose and (ep + 1) % 50 == 0:
                acc = self.fp32_accuracy(X, y)
                print(f"    epoch {ep+1:3d}  loss={ep_loss:.3f}  acc={acc:.3f}")

        return losses

    # ── Quantization-aware fine-tuning ─────────────────────────────────────────

    def qat_finetune(self, X: np.ndarray, y: np.ndarray,
                     bit_w: int, epochs: int = 40, lr: float = 0.004) -> float:
        """
        Fine-tune weights WITH simulated quantization (straight-through estimator).

        This is the "model adapts to hardware" half of the closed loop.
        After QAT, the same hardware config will give better inference accuracy,
        so the agent receives higher rewards for good hardware and keeps choosing it.

        Returns the improvement in FP32 accuracy (proxy for model quality).
        """
        clamp_max = (1 << (bit_w - 1)) - 1
        clamp_min = -(1 << (bit_w - 1))
        rng = np.random.default_rng(77)
        n   = len(X)

        def fake_quant(W):
            """Quantize → dequantize in FP32 (straight-through gradient)."""
            abs_max = float(np.abs(W).max())
            if abs_max == 0:
                return W
            sc = abs_max / clamp_max
            return np.clip(np.round(W / sc), clamp_min, clamp_max) * sc

        acc_before = self.fp32_accuracy(X, y)

        for _ in range(epochs):
            idx = rng.permutation(n)
            Xs, ys = X[idx], y[idx]

            for s in range(0, n, 64):
                Xb, yb = Xs[s:s+64], ys[s:s+64]
                nb = len(Xb)

                # Forward with quantized (fake-quant) weights
                W1f = fake_quant(self.W1)
                W2f = fake_quant(self.W2)
                H_pre = Xb @ W1f + self.b1
                H     = np.maximum(0.0, H_pre)
                logits = H @ W2f + self.b2
                probs  = self._softmax(logits)

                oh  = np.zeros_like(probs); oh[np.arange(nb), yb] = 1.0
                dZ2 = (probs - oh) / nb
                dW2 = H.T  @ dZ2;   db2 = dZ2.sum(0)
                dH  = dZ2  @ W2f.T
                dH1 = dH * (H_pre > 0)
                dW1 = Xb.T @ dH1;   db1 = dH1.sum(0)

                self.W1 -= lr * dW1;  self.b1 -= lr * db1
                self.W2 -= lr * dW2;  self.b2 -= lr * db2

        acc_after = self.fp32_accuracy(X, y)
        return acc_after - acc_before   # improvement delta

    # ── FPGA inference (the closed-loop measurement) ───────────────────────────

    def fpga_accuracy(self, X: np.ndarray, y: np.ndarray,
                      config: dict) -> dict:
        """
        Run inference using the FPGA's hardware arithmetic.

        Steps:
          1. Quantize inputs and weights to bit_w-bit integers
          2. Layer 1: vec_len-wide dot products + FPGA activation
          3. Layer 2: vec_len-wide dot products (no activation — argmax on logits)
          4. Compute classification accuracy vs ground truth

        This is where hardware meets model:
          • bit_w=4  → heavy quantization → accuracy drops
          • bit_w=8  → mild  quantization → accuracy ≈ FP32
          • bit_w=16 → near-lossless      → accuracy = FP32
          • act_type=relu → matches FP32 training → best accuracy
          • act_type=wrap → corrupts hidden units   → worst accuracy

        Returns
        -------
        dict with fpga_accuracy, fp32_accuracy, accuracy_ratio, quant_degradation
        """
        bit_w     = config["bit_w"]
        clamp_max = (1 << (bit_w - 1)) - 1
        clamp_min = -(1 << (bit_w - 1))

        # Quantize inputs
        abs_x   = float(np.abs(X).max())
        x_scale = abs_x / clamp_max if abs_x > 0 else 1.0
        X_q, _  = quantize(X, bit_w, scale=x_scale)

        # Quantize layer-1 weights / biases
        W1q, _ = quantize(self.W1, bit_w)
        b1q, _ = quantize(self.b1, bit_w)

        # Layer 1 forward (FPGA activation applied)
        H_q = fpga_layer(X_q, W1q, b1q, config, apply_act=True)

        # Re-quantize hidden activations to bit_w before feeding layer 2
        # (mirrors the output register of the FPGA pipeline)
        h_abs = float(np.abs(H_q).max())
        if h_abs > 0:
            h_scale   = h_abs / clamp_max
            H_q_rq    = np.clip(np.round(H_q / h_scale),
                                clamp_min, clamp_max).astype(np.int64)
        else:
            H_q_rq = H_q

        # Quantize layer-2 weights / biases
        W2q, _ = quantize(self.W2, bit_w)
        b2q, _ = quantize(self.b2, bit_w)

        # Layer 2 forward (no activation — use raw logits for argmax)
        cfg_linear = {**config, "act_type": 0}
        logits_q   = fpga_layer(H_q_rq, W2q, b2q, cfg_linear, apply_act=False)

        preds     = np.argmax(logits_q, axis=1)
        fpga_acc  = float((preds == y).mean())
        fp32_acc  = self.fp32_accuracy(X, y)

        return {
            "fpga_accuracy":      fpga_acc,
            "fp32_accuracy":      fp32_acc,
            "accuracy_ratio":     fpga_acc / max(fp32_acc, 1e-6),
            "quant_degradation":  fp32_acc - fpga_acc,
        }
