#!/usr/bin/env python3
"""
main_loop.py - AI-FPGA-Loop: Main orchestrator.

The AI agent continuously selects FPGA hardware design parameters,
generates synthesizable Verilog, evaluates performance, and learns to
improve over time using tabular Q-learning.

Usage:
    python main_loop.py --mode sim --iterations 20
    python main_loop.py --mode full --iterations 5 --review-every 2
"""

import argparse
import json
import os
import sys
import time
import math
from datetime import datetime

# Project modules
from design_agent import FPGADesignAgent, TOTAL_CONFIGS
from verilog_gen   import generate_verilog, save_best_verilog
from simulator     import run_full_simulation
from reward        import compute_reward, breakdown

# ── ANSI colour helpers ───────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
GREY   = "\033[90m"
MAGENTA = "\033[95m"


def c(text, *codes):
    return "".join(codes) + str(text) + RESET


def _banner():
    print(c("""
╔══════════════════════════════════════════════════════════════════╗
║          AI-FPGA-LOOP  ·  Autonomous Hardware Design             ║
║  AI selects → Verilog generated → Synthesized → Evaluated → RL  ║
╚══════════════════════════════════════════════════════════════════╝
""", CYAN, BOLD))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI-FPGA-Loop: continuous AI-driven FPGA design optimisation"
    )
    p.add_argument("--mode",         default="sim",
                   choices=["sim", "full"],
                   help="Run mode: sim (Python only) or full (Yosys+iverilog)")
    p.add_argument("--iterations",   type=int, default=20,
                   help="Number of design iterations")
    p.add_argument("--review-every", type=int, default=10,
                   help="Pause for human review every N iterations")
    p.add_argument("--target",       default="ops_per_lut",
                   choices=["ops_per_lut", "throughput", "frequency", "latency"],
                   help="Primary optimisation target for display")
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress verbose per-iteration output")
    p.add_argument("--lr",           type=float, default=0.1,
                   help="Q-learning learning rate")
    p.add_argument("--epsilon",      type=float, default=0.9,
                   help="Initial epsilon for ε-greedy exploration")
    return p.parse_args()


# ── Human review checkpoint ───────────────────────────────────────────────────

def human_review(agent: FPGADesignAgent, iteration: int, args):
    print()
    print(c("━" * 68, YELLOW))
    print(c(f"  HUMAN REVIEW CHECKPOINT  —  after iteration {iteration}", YELLOW, BOLD))
    print(c("━" * 68, YELLOW))

    s = agent.stats()
    print(f"\n  Configs explored : {s['configs_explored']} / {TOTAL_CONFIGS}"
          f"  ({s['coverage_pct']}%)")
    print(f"  Best reward      : {c(s['best_reward'], GREEN, BOLD)}")
    print(f"  Epsilon          : {s['epsilon']:.3f}")
    print(f"  Reward trend     : {s['avg_reward_first10']:.4f} → "
          f"{s['avg_reward_last10']:.4f}  "
          f"{c(agent.reward_trend_arrow(), GREEN if s['reward_trend']=='up' else RED)}")

    top = agent.top_designs(3)
    print(f"\n  Top-3 designs by Q-value:")
    for i, d in enumerate(top):
        cfg = d["config"]
        print(f"    {i+1}. Q={c(d['q_value'], CYAN)}  "
              f"bit_w={cfg['bit_w']} vec_len={cfg['vec_len']} "
              f"pipe={cfg['pipe_stages']} act={cfg['act_type']} "
              f"dsp={cfg['use_dsp']}")

    insights = agent.improvement_analysis()
    if insights:
        print(f"\n  AI Improvement Insights:")
        for param, info in insights.items():
            pref = info["preferred_values"]
            if pref:
                vals = ", ".join(f"{v}(+{d:.0%})" for v, d in pref[:2])
                print(f"    {param:12s} prefers: {c(vals, GREEN)}")

    print()
    print(c("  Commands: [C]ontinue  [S]top  [R]eset exploration  "
            "[V]iew last 5  [H]istory", GREY))
    try:
        cmd = input("  > ").strip().upper()
    except EOFError:
        cmd = "C"

    if cmd == "S":
        print(c("  Stopping at user request.", YELLOW))
        return False
    elif cmd == "R":
        agent.reset_exploration(0.5)
        print(c("  Epsilon reset to 0.5 — resuming exploration.", YELLOW))
    elif cmd == "V":
        last5 = agent.last_n(5)
        print(f"\n  Last 5 iterations:")
        for e in last5:
            cfg = e["config"]
            print(f"    reward={e['reward']:.4f}  bit_w={cfg['bit_w']} "
                  f"vec_len={cfg['vec_len']} pipe={cfg['pipe_stages']}")
        input("  [Enter to continue]")
    elif cmd == "H":
        st = agent.stats()
        arrow = agent.reward_trend_arrow()
        print(f"\n  History stats:")
        print(f"    Total iterations : {st['iterations']}")
        print(f"    Avg first 10     : {st['avg_reward_first10']:.4f}")
        print(f"    Avg last 10      : {st['avg_reward_last10']:.4f}  {arrow}")
        print(f"    Delta            : {st['reward_trend_delta']:+.4f}")
        input("  [Enter to continue]")

    print(c("  Resuming loop...", GREY))
    print()
    return True


# ── Per-iteration print ───────────────────────────────────────────────────────

def _iter_print(it: int, method: str, config: dict, metrics: dict,
                reward: float, is_best: bool, quiet: bool):
    if quiet and not is_best:
        return

    method_col = GREEN if method == "exploit" else BLUE
    flag = c(" ★ NEW BEST", YELLOW, BOLD) if is_best else ""

    cfg_str = (f"bit_w={config['bit_w']:2d} vec={config['vec_len']:2d} "
               f"pipe={config['pipe_stages']} act={config['act_type']} "
               f"dsp={config['use_dsp']} ext={config['accum_extra']}")

    luts  = metrics.get("luts", 0)
    freq  = metrics.get("frequency", 0.0)
    tput  = metrics.get("throughput", 0.0)
    mae   = metrics.get("mae", 0.0)
    opl   = metrics.get("ops_per_lut", 0.0)

    rew_col = GREEN if reward > 0.5 else (YELLOW if reward > 0 else RED)

    print(
        f"  {c(f'[{it:04d}]', GREY)} "
        f"{c(f'{method:7s}', method_col)}  "
        f"{cfg_str}  "
        f"LUTs={luts:5d}  "
        f"F={freq:6.1f}MHz  "
        f"T={tput/1e6:7.1f}MOPS  "
        f"MAE={mae:.4f}  "
        f"R={c(f'{reward:+.4f}', rew_col)}"
        f"{flag}"
    )


# ── Final report ──────────────────────────────────────────────────────────────

def _final_report(agent: FPGADesignAgent, best_metrics: dict,
                  elapsed: float, n_iter: int):
    print()
    print(c("═" * 68, CYAN, BOLD))
    print(c("  FINAL REPORT", CYAN, BOLD))
    print(c("═" * 68, CYAN, BOLD))

    s = agent.stats()
    print(f"\n  Total iterations : {n_iter}")
    print(f"  Elapsed time     : {elapsed:.1f}s  "
          f"({elapsed/max(n_iter,1):.2f}s / iter)")
    print(f"  Configs explored : {s['configs_explored']} / {TOTAL_CONFIGS}  "
          f"({s['coverage_pct']}%)")
    print(f"  Reward trend     : {agent.reward_trend_arrow()}  "
          f"({s['avg_reward_first10']:.4f} → {s['avg_reward_last10']:.4f})")

    best = agent.best_entry
    print(f"\n{c('  BEST DESIGN', GREEN, BOLD)}")
    if best:
        cfg = best["config"]
        print(f"    Config  : bit_w={cfg['bit_w']} vec_len={cfg['vec_len']} "
              f"pipe={cfg['pipe_stages']} act={cfg['act_type']} "
              f"dsp={cfg['use_dsp']} ext={cfg['accum_extra']}")
        print(f"    Reward  : {c(best['reward'], GREEN, BOLD)}")
        if best_metrics:
            print(f"    LUTs    : {best_metrics.get('luts', '?')}")
            print(f"    Freq    : {best_metrics.get('frequency', '?')} MHz")
            print(f"    Tput    : {best_metrics.get('throughput', 0)/1e6:.1f} MOPS")
            print(f"    MAE     : {best_metrics.get('mae', '?'):.4f}")
            print(f"    OPS/LUT : {best_metrics.get('ops_per_lut', 0):.0f}")

    top5 = agent.top_designs(5)
    print(f"\n{c('  TOP-5 DESIGNS (by Q-value)', CYAN)}")
    for i, d in enumerate(top5):
        cfg = d["config"]
        print(f"    {i+1}. Q={d['q_value']:+.4f}  "
              f"bit_w={cfg['bit_w']} vec={cfg['vec_len']} "
              f"pipe={cfg['pipe_stages']} act={cfg['act_type']} "
              f"dsp={cfg['use_dsp']}")

    insights = agent.improvement_analysis()
    if insights:
        print(f"\n{c('  AI IMPROVEMENT INSIGHTS', MAGENTA)}")
        for param, info in insights.items():
            pref = info["preferred_values"]
            if pref:
                vals = ", ".join(f"{v}(+{d:.0%})" for v, d in pref[:3])
                print(f"    {param:12s} →  {c(vals, GREEN)}")

    print(c("\n═" * 68 + "\n", CYAN))


# ── JSON log ──────────────────────────────────────────────────────────────────

def _save_log(agent: FPGADesignAgent, run_info: dict, best_metrics: dict,
              log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"run_{ts}.json")

    # Serialise history (ensure JSON-safe types)
    history = []
    for e in agent.history:
        history.append({
            "config": {k: int(v) if isinstance(v, (int, float)) else v
                       for k, v in e["config"].items()},
            "reward": float(e["reward"]),
        })

    payload = {
        "run_info":   run_info,
        "best_design": {
            "config":  agent.best_entry.get("config", {}),
            "reward":  float(agent.best_reward),
            "metrics": {k: (float(v) if isinstance(v, float) else v)
                        for k, v in best_metrics.items()},
        },
        "agent_stats": agent.stats(),
        "history":     history,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.quiet:
        _banner()
        print(f"  Mode        : {c(args.mode.upper(), CYAN, BOLD)}")
        print(f"  Iterations  : {args.iterations}")
        print(f"  Review every: {args.review_every}")
        print(f"  LR / ε      : {args.lr} / {args.epsilon}")
        print()

    # Full-mode tool check
    if args.mode == "full":
        from synthesizer import check_tools_available, run_full_synthesis
        tools = check_tools_available()
        if not tools["all_ok"]:
            print(c("  WARNING: Yosys or iverilog not found — falling back to sim mode.",
                    YELLOW))
            args.mode = "sim"
        else:
            print(c(f"  Full mode: yosys={tools['yosys']}  "
                    f"iverilog={tools['iverilog']}", GREEN))
            print()

    agent = FPGADesignAgent(
        lr=args.lr,
        epsilon=args.epsilon,
    )

    best_metrics: dict = {}
    start_time = time.time()

    run_info = {
        "mode":          args.mode,
        "iterations":    args.iterations,
        "review_every":  args.review_every,
        "started_at":    datetime.utcnow().isoformat(),
        "target":        args.target,
    }

    try:
        for it in range(1, args.iterations + 1):

            # 1. AI selects config
            config, method = agent.select_config()

            # 2. Generate Verilog
            verilog_path = generate_verilog(config, iteration=it,
                                            build_dir="build")

            # 3. Evaluate
            if args.mode == "full":
                from synthesizer import run_full_synthesis
                metrics = run_full_synthesis(verilog_path, config,
                                             work_dir="build")
            else:
                metrics = run_full_simulation(config, quiet=args.quiet)

            # 4. Compute reward
            reward = compute_reward(metrics)

            # 5. Update agent
            agent.update(config, reward)

            is_best = (reward >= agent.best_reward and
                       abs(reward - agent.best_reward) < 1e-9)
            if is_best:
                best_metrics = metrics.copy()

            # 6. Console output
            _iter_print(it, method, config, metrics, reward,
                        is_best, args.quiet)

            # 7. Human review checkpoint
            if it % args.review_every == 0 and it < args.iterations:
                should_continue = human_review(agent, it, args)
                if not should_continue:
                    break

    except KeyboardInterrupt:
        print(c("\n  Interrupted by user.", YELLOW))

    elapsed = time.time() - start_time

    # Save best Verilog
    if agent.best_entry:
        best_path = save_best_verilog(agent.best_entry["config"], build_dir="build")
        print(c(f"\n  Best Verilog saved → {best_path}", GREEN))

    # Final report
    _final_report(agent, best_metrics, elapsed, len(agent.history))

    # Save JSON log
    log_path = _save_log(agent, run_info, best_metrics)
    print(c(f"  JSON log saved  → {log_path}", GREEN))


if __name__ == "__main__":
    main()
