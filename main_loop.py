#!/usr/bin/env python3
"""
main_loop.py - AI-FPGA Closed Loop Orchestrator.

THE CLOSED LOOP
---------------
  1. AI agent (DQN) selects FPGA hardware parameters
  2. Synthesizable Verilog is generated for a dot-product unit
  3. The hardware is evaluated (sim or Yosys):
       - Hardware metrics: LUTs, FFs, frequency, throughput
       - NN inference on the hardware: bit-exact fixed-point forward pass
  4. Inference accuracy drives the reward (40% weight — primary signal)
  5. Agent learns → proposes better hardware next iteration
  6. Every K iters: QAT fine-tunes the NN for the current best hardware
       → model adapts to hardware constraints → accuracy improves
       → higher reward for same hardware → agent keeps selecting it

  Hardware improves → model runs better → reward rises → agent locks in good HW
  Model fine-tunes → same HW gives even better accuracy → loop converges

Usage
-----
  python main_loop.py --mode sim --iterations 30
  python main_loop.py --mode full --iterations 10 --review-every 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from design_agent import FPGADesignAgent, TOTAL_CONFIGS
from verilog_gen   import generate_verilog, save_best_verilog
from simulator     import run_full_simulation
from reward        import compute_reward, breakdown
from dataset       import make_dataset
from neural_net    import TinyMLP

# ── ANSI colours ───────────────────────────────────────────────────────────────
R = "\033[0m";  B = "\033[1m"
RED = "\033[91m";  GRN = "\033[92m";  YLW = "\033[93m"
BLU = "\033[94m";  CYN = "\033[96m";  GRY = "\033[90m";  MAG = "\033[95m"

def c(t, *codes): return "".join(codes) + str(t) + R


def _banner():
    print(c("""
╔══════════════════════════════════════════════════════════════════════╗
║          AI-FPGA-LOOP  ·  Hardware/Model Co-Design Closed Loop       ║
║  AI designs FPGA  →  FPGA runs NN inference  →  accuracy feeds back  ║
╚══════════════════════════════════════════════════════════════════════╝
""", CYN, B))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI-FPGA Closed Loop: AI co-designs hardware and inference model"
    )
    p.add_argument("--mode",          default="sim", choices=["sim", "full"])
    p.add_argument("--iterations",    type=int,   default=30)
    p.add_argument("--review-every",  type=int,   default=10)
    p.add_argument("--finetune-every",type=int,   default=10,
                   help="QAT fine-tune the NN every N iterations")
    p.add_argument("--target",        default="inference_accuracy",
                   choices=["inference_accuracy", "ops_per_lut", "throughput"])
    p.add_argument("--quiet",         action="store_true")
    p.add_argument("--lr",            type=float, default=0.1)
    p.add_argument("--epsilon",       type=float, default=0.9)
    p.add_argument("--train-epochs",  type=int,   default=250,
                   help="FP32 training epochs for the NN before the loop")
    return p.parse_args()


# ── Model initialisation ───────────────────────────────────────────────────────

def init_model(train_epochs: int, quiet: bool):
    """Train the FP32 baseline model and return (model, X_tr, y_tr, X_te, y_te)."""
    if not quiet:
        print(c("  Initialising dataset and training FP32 baseline model...", GRY))

    X_tr, y_tr, X_te, y_te, meta = make_dataset()
    model = TinyMLP()
    model.train(X_tr, y_tr, epochs=train_epochs, verbose=not quiet)

    fp32_acc = model.fp32_accuracy(X_te, y_te)
    if not quiet:
        print(c(f"  FP32 baseline accuracy: {fp32_acc*100:.1f}%"
                f"  (test set, {len(y_te)} samples)", GRN, B))
        print()

    return model, X_tr, y_tr, X_te, y_te, fp32_acc


# ── Human review ───────────────────────────────────────────────────────────────

def human_review(agent: FPGADesignAgent, iteration: int,
                 model: TinyMLP, X_te, y_te):
    print()
    print(c("━" * 70, YLW))
    print(c(f"  HUMAN REVIEW CHECKPOINT  —  after iteration {iteration}", YLW, B))
    print(c("━" * 70, YLW))

    s = agent.stats()
    print(f"\n  Configs explored : {s['configs_explored']} / {TOTAL_CONFIGS}"
          f"  ({s['coverage_pct']}%)")
    print(f"  Best reward      : {c(s['best_reward'], GRN, B)}")
    print(f"  Epsilon          : {s['epsilon']:.3f}")
    print(f"  Reward trend     : {s['avg_reward_first10']:.4f} → "
          f"{s['avg_reward_last10']:.4f}  "
          f"{c(agent.reward_trend_arrow(), GRN if s['reward_trend']=='up' else RED)}")

    # Show current model accuracy on best hardware
    if agent.best_entry:
        bc  = agent.best_entry["config"]
        res = model.fpga_accuracy(X_te, y_te, bc)
        print(f"\n  Best HW inference accuracy:")
        print(f"    FP32 baseline : {res['fp32_accuracy']*100:.1f}%")
        print(f"    FPGA (bit_w={bc['bit_w']}) : "
              f"{c(f\"{res['fpga_accuracy']*100:.1f}%\", GRN)}")
        print(f"    Degradation   : {res['quant_degradation']*100:.1f}%")

    top = agent.top_designs(3)
    print(f"\n  Top-3 designs by Q-value:")
    for i, d in enumerate(top):
        cfg = d["config"]
        print(f"    {i+1}. Q={c(d['q_value'], CYN)}  "
              f"bit_w={cfg['bit_w']} vec={cfg['vec_len']} "
              f"pipe={cfg['pipe_stages']} act={cfg['act_type']} dsp={cfg['use_dsp']}")

    insights = agent.improvement_analysis()
    if insights:
        print(f"\n  AI Insights (top-k vs bottom-k param preferences):")
        for param, info in insights.items():
            pref = info["preferred_values"]
            if pref:
                vals = ", ".join(f"{v}(+{d:.0%})" for v, d in pref[:2])
                print(f"    {param:12s} prefers: {c(vals, GRN)}")

    print()
    print(c("  [C]ontinue  [S]top  [R]eset exploration  [V]iew last 5  [H]istory",
            GRY))
    try:
        cmd = input("  > ").strip().upper()
    except EOFError:
        cmd = "C"

    if cmd == "S":
        print(c("  Stopping at user request.", YLW)); return False
    elif cmd == "R":
        agent.reset_exploration(0.5)
        print(c("  Epsilon reset to 0.5.", YLW))
    elif cmd == "V":
        for e in agent.last_n(5):
            cfg = e["config"]
            print(f"    r={e['reward']:.4f}  bit_w={cfg['bit_w']} "
                  f"vec={cfg['vec_len']} pipe={cfg['pipe_stages']}")
        input("  [Enter] ")
    elif cmd == "H":
        st = agent.stats()
        print(f"\n    {st['iterations']} iters  "
              f"first10={st['avg_reward_first10']:.4f}  "
              f"last10={st['avg_reward_last10']:.4f}  "
              f"{agent.reward_trend_arrow()}")
        input("  [Enter] ")

    print(c("  Resuming loop...\n", GRY))
    return True


# ── Per-iteration print ────────────────────────────────────────────────────────

def _iter_print(it, method, cfg, metrics, reward, is_best, quiet):
    if quiet and not is_best:
        return

    mc  = GRN if method == "exploit" else BLU
    flg = c("  ★ NEW BEST", YLW, B) if is_best else ""

    fps = metrics.get("fp32_accuracy",      0.0) * 100
    fpa = metrics.get("inference_accuracy", 0.0) * 100
    deg = (fps - fpa)
    deg_col = GRN if deg < 5 else (YLW if deg < 15 else RED)

    rc  = GRN if reward > 0.6 else (YLW if reward > 0.3 else RED)
    luts = metrics.get("luts", 0)
    freq = metrics.get("frequency", 0.0)

    print(
        f"  {c(f'[{it:04d}]', GRY)} {c(f'{method:7s}', mc)}  "
        f"b={cfg['bit_w']:2d} v={cfg['vec_len']:2d} p={cfg['pipe_stages']} "
        f"act={cfg['act_type']} dsp={cfg['use_dsp']}  "
        f"LUTs={luts:4d}  {freq:5.0f}MHz  "
        f"FP32={fps:4.1f}% → "
        f"FPGA={c(f'{fpa:4.1f}%', GRN)}  "
        f"Δ={c(f'-{deg:.1f}%', deg_col)}  "
        f"R={c(f'{reward:+.4f}', rc)}"
        f"{flg}"
    )


# ── QAT event print ────────────────────────────────────────────────────────────

def _qat_print(bit_w, delta_acc, quiet):
    if quiet and abs(delta_acc) < 0.005:
        return
    col   = GRN if delta_acc >= 0 else RED
    arrow = "↑" if delta_acc >= 0 else "↓"
    print(c(f"\n  ═══ QAT fine-tune (bit_w={bit_w}) "
            f"model accuracy {arrow} {delta_acc*100:+.1f}% "
            f"[hardware→model feedback] ═══\n", MAG))


# ── Final report ───────────────────────────────────────────────────────────────

def _final_report(agent, best_metrics, fp32_acc, elapsed, n_iter):
    print()
    print(c("═" * 70, CYN, B))
    print(c("  FINAL REPORT — AI-FPGA Closed Loop", CYN, B))
    print(c("═" * 70, CYN, B))

    s = agent.stats()
    print(f"\n  Iterations    : {n_iter}")
    print(f"  Elapsed       : {elapsed:.1f}s  ({elapsed/max(n_iter,1):.2f}s/iter)")
    print(f"  Coverage      : {s['configs_explored']} / {TOTAL_CONFIGS} "
          f"({s['coverage_pct']}%)")
    print(f"  Reward trend  : {agent.reward_trend_arrow()}  "
          f"({s['avg_reward_first10']:.4f} → {s['avg_reward_last10']:.4f})")

    be = agent.best_entry
    if be:
        cfg = be["config"]
        print(f"\n{c('  BEST HARDWARE CONFIG', GRN, B)}")
        print(f"    bit_w={cfg['bit_w']}  vec_len={cfg['vec_len']}  "
              f"pipe={cfg['pipe_stages']}  act={cfg['act_type']}  "
              f"dsp={cfg['use_dsp']}  accum_extra={cfg['accum_extra']}")
        print(f"    Reward : {c(be['reward'], GRN, B)}")

        if best_metrics:
            fpa  = best_metrics.get("inference_accuracy", 0) * 100
            fps  = best_metrics.get("fp32_accuracy", fp32_acc) * 100
            luts = best_metrics.get("luts", 0)
            freq = best_metrics.get("frequency", 0)
            tput = best_metrics.get("throughput", 0) / 1e6
            print(f"\n{c('  INFERENCE QUALITY', CYN)}")
            print(f"    FP32 baseline  : {fps:.1f}%")
            print(f"    FPGA accuracy  : {c(f'{fpa:.1f}%', GRN, B)}")
            print(f"    Degradation    : {fps-fpa:.1f}%  "
                  f"({fpa/max(fps,0.01)*100:.0f}% of FP32 retained)")
            print(f"\n{c('  HARDWARE EFFICIENCY', CYN)}")
            print(f"    LUTs           : {luts}")
            print(f"    Clock          : {freq:.0f} MHz")
            print(f"    Throughput     : {tput:.0f} MOPS")
            print(f"    OPS/LUT        : {best_metrics.get('ops_per_lut',0):.0f}")

    top5 = agent.top_designs(5)
    print(f"\n{c('  TOP-5 HARDWARE DESIGNS (by Q-value)', CYN)}")
    for i, d in enumerate(top5):
        cfg = d["config"]
        print(f"    {i+1}. Q={d['q_value']:+.4f}  "
              f"bit_w={cfg['bit_w']} vec={cfg['vec_len']} "
              f"pipe={cfg['pipe_stages']} act={cfg['act_type']} dsp={cfg['use_dsp']}")

    insights = agent.improvement_analysis()
    if insights:
        print(f"\n{c('  AI INSIGHTS: hardware params that help inference', MAG)}")
        for param, info in insights.items():
            pref = info["preferred_values"]
            if pref:
                vals = ", ".join(f"{v}(+{d:.0%})" for v, d in pref[:3])
                print(f"    {param:12s} →  {c(vals, GRN)}")

    print(c("\n" + "═"*70 + "\n", CYN))


# ── JSON log ───────────────────────────────────────────────────────────────────

def _save_log(agent, run_info, best_metrics, fp32_acc, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"run_{ts}.json")

    history = [
        {"config": {k: int(v) if isinstance(v, (int, float)) else v
                    for k, v in e["config"].items()},
         "reward": float(e["reward"])}
        for e in agent.history
    ]

    payload = {
        "run_info":      run_info,
        "fp32_baseline": fp32_acc,
        "best_design": {
            "config":  agent.best_entry.get("config", {}) if agent.best_entry else {},
            "reward":  float(agent.best_reward),
            "metrics": {k: float(v) if isinstance(v, float) else v
                        for k, v in best_metrics.items()},
        },
        "agent_stats": agent.stats(),
        "history":     history,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()

    if not args.quiet:
        _banner()
        print(f"  Mode           : {c(args.mode.upper(), CYN, B)}")
        print(f"  Iterations     : {args.iterations}")
        print(f"  Review every   : {args.review_every}")
        print(f"  QAT every      : {args.finetune_every} iters")
        print(f"  LR / ε         : {args.lr} / {args.epsilon}")
        print()

    if args.mode == "full":
        from synthesizer import check_tools_available
        tools = check_tools_available()
        if not tools["all_ok"]:
            print(c("  WARNING: Yosys/iverilog not found — falling back to sim.", YLW))
            args.mode = "sim"

    # ── Phase 1: train baseline model ─────────────────────────────────────────
    model, X_tr, y_tr, X_te, y_te, fp32_acc = init_model(
        args.train_epochs, args.quiet
    )

    # ── Phase 2: DQN loop ─────────────────────────────────────────────────────
    agent = FPGADesignAgent(lr=args.lr, epsilon=args.epsilon)

    best_metrics: dict = {}
    start_time = time.time()

    run_info = {
        "mode":          args.mode,
        "iterations":    args.iterations,
        "review_every":  args.review_every,
        "finetune_every": args.finetune_every,
        "started_at":    datetime.utcnow().isoformat(),
        "fp32_baseline": fp32_acc,
    }

    try:
        for it in range(1, args.iterations + 1):

            # 1. Agent selects hardware config
            config, method = agent.select_config()

            # 2. Generate Verilog
            verilog_path = generate_verilog(config, iteration=it, build_dir="build")

            # 3. Hardware metrics (sim mode or Yosys)
            if args.mode == "full":
                from synthesizer import run_full_synthesis
                hw = run_full_synthesis(verilog_path, config, work_dir="build")
            else:
                hw = run_full_simulation(config, quiet=True)

            # 4. NN inference on this hardware  ← THE CLOSED LOOP MEASUREMENT
            nn_result = model.fpga_accuracy(X_te, y_te, config)

            # 5. Assemble full metrics dict
            metrics = {
                **hw,
                "inference_accuracy": nn_result["fpga_accuracy"],
                "fp32_accuracy":      nn_result["fp32_accuracy"],
                "accuracy_ratio":     nn_result["accuracy_ratio"],
                "quant_degradation":  nn_result["quant_degradation"],
                # keep hw_pass_rate for the hard gate
                "hw_pass_rate":       hw.get("pass_rate", 1.0),
            }

            # 6. Reward: 40% inference accuracy + 60% hardware efficiency
            reward = compute_reward(metrics)

            # 7. Update agent
            agent.update(config, reward)

            is_best = (reward >= agent.best_reward and
                       abs(reward - agent.best_reward) < 1e-9)
            if is_best:
                best_metrics = metrics.copy()

            # 8. Print
            _iter_print(it, method, config, metrics, reward, is_best, args.quiet)

            # 9. QAT fine-tuning  ← MODEL ADAPTS TO HARDWARE
            if it % args.finetune_every == 0 and agent.best_entry:
                best_hw_cfg = agent.best_entry["config"]
                delta = model.qat_finetune(
                    X_tr, y_tr,
                    bit_w=best_hw_cfg["bit_w"],
                    epochs=40,
                )
                _qat_print(best_hw_cfg["bit_w"], delta, args.quiet)

                # Re-measure best hardware accuracy after model improvement
                if is_best or True:
                    updated = model.fpga_accuracy(X_te, y_te, best_hw_cfg)
                    best_metrics["inference_accuracy"] = updated["fpga_accuracy"]
                    best_metrics["fp32_accuracy"]      = updated["fp32_accuracy"]

            # 10. Human review checkpoint
            if it % args.review_every == 0 and it < args.iterations:
                go = human_review(agent, it, model, X_te, y_te)
                if not go:
                    break

    except KeyboardInterrupt:
        print(c("\n  Interrupted.", YLW))

    elapsed = time.time() - start_time

    # Save best Verilog
    if agent.best_entry:
        best_path = save_best_verilog(agent.best_entry["config"], build_dir="build")
        print(c(f"\n  Best Verilog → {best_path}", GRN))

    # Final report
    _final_report(agent, best_metrics, fp32_acc, elapsed, len(agent.history))

    # Save log
    log_path = _save_log(agent, run_info, best_metrics, fp32_acc)
    print(c(f"  JSON log     → {log_path}", GRN))


if __name__ == "__main__":
    main()
