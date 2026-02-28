# AI-FPGA-Loop

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-proof--of--concept-orange)]()
[![NumPy](https://img.shields.io/badge/deps-numpy-blue)]()

> **An AI agent that autonomously designs FPGA inference hardware, measures its
> performance, and iterates to improve it in a closed loop.**

---

## What It Does

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          AI-FPGA-LOOP                                    │
│                                                                          │
│  ┌───────────┐    ┌─────────────┐    ┌──────────────────────────────┐   │
│  │  Q-Agent  │───▶│ Verilog Gen │───▶│  Synthesizer / Simulator     │   │
│  │ ε-greedy  │    │ dot_product │    │  • LUT / FF counts           │   │
│  │ Q-table   │    │ _unit.v     │    │  • Clock frequency           │   │
│  └─────┲─────┘    └─────────────┘    │  • Throughput                │   │
│        ┃                             └──────────────┬───────────────┘   │
│        ┃                                            │ HW config         │
│        ┃          ┌──────────────────────────────┐  │                   │
│        ┃          │  TinyMLP Inference (8→16→4)  │◀─┘                   │
│        ┃          │  • Quantized FPGA inference  │                      │
│        ┃          │  • FP32 baseline accuracy    │                      │
│        ┃          │  • QAT fine-tuning (×10 iters│                      │
│        ┃          └──────────────┬───────────────┘                      │
│        ┃                         │ inference_acc / fp32_acc             │
│        ┃          ┌──────────────▼───────────────┐                      │
│        ┗━━━━━━━━━━│  Reward Fn (40% infer. acc)  │                      │
│                   └──────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────────────┘
```

The **FPGA hardware being designed** is a parameterised dot-product unit:

```
y = activation( Σ w_i · x_i + bias )
```

This is the fundamental compute primitive inside every neural network layer.
The twist: a real **TinyMLP (8→16→4)** runs inference *on that hardware*
using its quantization and activation settings. How much FP32 accuracy the
quantized FPGA hardware retains is the primary closed-loop signal.

Every 10 iterations, **QAT (quantization-aware training)** fine-tunes the
model for the current best hardware config — so the model learns to tolerate
the hardware's constraints, which raises the reward for good configs and
reinforces the agent's preference for them.

---

## Design Space (648 configurations)

| Parameter      | Options          | Trade-off                        |
|----------------|------------------|----------------------------------|
| `bit_w`        | 4, 8, 16         | precision vs. resource usage     |
| `vec_len`      | 2, 4, 8, 16      | parallelism vs. area             |
| `pipe_stages`  | 1, 2, 3          | throughput vs. latency           |
| `act_type`     | wrap, relu, clamp| accuracy vs. simplicity          |
| `accum_extra`  | 2, 4, 8          | overflow safety vs. width        |
| `use_dsp`      | LUT-mult, DSP    | efficiency vs. portability       |

---

## Performance Metrics & Reward

The reward function mirrors `reward.py` exactly.

**Hard gates** (checked before weighted scoring):

| Condition                         | Reward |
|-----------------------------------|--------|
| Synthesis failure                 | −1.0   |
| HW pass rate < 95%                | −0.8   |

**Weighted score** (applied only when both gates pass):

| Metric                              | Direction | Weight | Normalisation target |
|-------------------------------------|-----------|--------|----------------------|
| Inference acc (FPGA ÷ FP32)         | ↑ higher  | **40%**| ratio = 1.0          |
| Throughput (OPS/s)                  | ↑ higher  | 25%    | 2 GOPS               |
| LUT resource usage                  | ↓ fewer   | 20%    | 500 LUTs             |
| Clock frequency                     | ↑ higher  | 10%    | 200 MHz              |
| Pipeline latency (cycles)           | ↓ lower   | 5%     | 3 cycles             |

The **inference accuracy score** is the fraction of FP32 accuracy retained on
the quantized FPGA hardware:

```
s_inf = fpga_accuracy / fp32_accuracy     # clamped to [0, 1]

reward = 0.40·s_inf + 0.25·s_tput + 0.20·s_res + 0.10·s_freq + 0.05·s_lat
```

- `bit_w=4 + act=wrap` → heavy quantization → accuracy collapses → low reward
- `bit_w=8 + act=relu` → mild quantization → accuracy preserved → high reward
- `bit_w=16 + act=relu` → near-lossless → best accuracy → highest reward

---

## Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/ai-fpga-loop.git
cd ai-fpga-loop

# Python deps (standard lib + numpy only)
pip install numpy

# Optional: for --mode full
# macOS:  brew install yosys icarus-verilog
# Ubuntu: sudo apt install yosys iverilog
```

---

## Usage

```bash
# Fast simulation mode (no external tools needed)
python main_loop.py --mode sim --iterations 20

# Quiet mode
python main_loop.py --mode sim --iterations 50 --quiet

# Full synthesis mode (requires yosys + iverilog)
python main_loop.py --mode full --iterations 10

# Custom review interval and learning rate
python main_loop.py --mode sim --iterations 100 --review-every 20 --lr 0.05

# Primary optimisation target display
python main_loop.py --mode sim --iterations 30 --target ops_per_lut
```

### CLI Flags

| Flag            | Default       | Description                              |
|-----------------|---------------|------------------------------------------|
| `--mode`        | `sim`         | `sim` (Python) or `full` (Yosys)         |
| `--iterations`  | 20            | Number of design–evaluate cycles         |
| `--review-every`| 10            | Human checkpoint every N iterations      |
| `--target`      | `ops_per_lut` | Primary display metric                   |
| `--quiet`       | off           | Only print new-best iterations           |
| `--lr`          | 0.1           | Q-learning learning rate                 |
| `--epsilon`     | 0.9           | Initial ε for exploration                |

---

## Human Review Checkpoints

Every `--review-every` iterations the loop pauses and shows:

- Current exploration coverage (N/648 configs)
- Top-3 designs found so far
- AI improvement insights (which parameter values correlate with better results)
- Reward trend (first-10 avg vs. last-10 avg with ↑/↓)

Interactive commands:

| Key | Action                                     |
|-----|--------------------------------------------|
| C   | Continue the loop                          |
| S   | Stop and show final report                 |
| R   | Reset epsilon to 0.5 (more exploration)    |
| V   | View last 5 iterations                     |
| H   | History statistics with reward trend       |

---

## Output Files

```
build/
  iter_0001_dot_product_unit.v   # Verilog for each iteration
  iter_0002_dot_product_unit.v
  ...
  BEST_DESIGN.v                  # Best design found (saved at end)

logs/
  run_20240101_120000.json       # Complete run log with full history
```

---

## Project Structure

```
ai-fpga-loop/
├── main_loop.py       # Main orchestrator & CLI
├── design_agent.py    # Tabular Q-learning agent
├── neural_net.py      # TinyMLP (8→16→4): FP32 training, FPGA inference, QAT
├── dataset.py         # Synthetic 4-class/8-feature classification benchmark
├── verilog_gen.py     # Synthesizable Verilog generator
├── simulator.py       # Pure-Python fixed-point dot-product simulator
├── synthesizer.py     # Yosys + iverilog wrappers
├── reward.py          # Reward function (40% inference acc + 60% HW metrics)
├── index.html         # Browser-based interactive simulator (self-contained)
├── upload_to_github.sh
├── build/             # Generated Verilog files
├── logs/              # JSON run logs
└── README.md
```

---

## How the AI Works

The agent uses **tabular Q-learning** over the 648-configuration design space:

1. **ε-greedy selection** — with probability ε explore a random (preferably
   unseen) config; otherwise exploit the config with the highest Q-value.
2. **Q-table update** — `Q(s) ← (1−α)·Q(s) + α·(r + γ·max Q)`
3. **Epsilon decay** — ε decays by 0.97 per iteration toward a minimum of 0.05,
   shifting from exploration toward exploitation.
4. **Improvement analysis** — at checkpoints the agent compares which parameter
   values appear more often in the top-k vs bottom-k designs.

---

## Generated Verilog Features

- `genvar`-based parallel multiplier array
- Signed adder tree with proper sign extension
- Pipeline registers with async active-low reset
- Combinational activation (wrap / ReLU / clamp)
- DSP or LUT inference hint via synthesis attributes

---

## License

MIT — see [LICENSE](LICENSE).
