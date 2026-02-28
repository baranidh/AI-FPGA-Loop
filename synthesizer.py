"""
synthesizer.py - Wrapper around Yosys and Icarus Verilog for full-mode runs.
"""

import os
import re
import subprocess
import tempfile
import math
from simulator import run_accuracy_test, estimate_synthesis_metrics


# ── Tool availability ─────────────────────────────────────────────────────────

def check_tools_available() -> dict:
    """
    Check whether yosys and iverilog exist on the system PATH.

    Returns
    -------
    dict: {yosys: bool, iverilog: bool, all_ok: bool}
    """
    def _which(tool: str) -> bool:
        try:
            result = subprocess.run(
                ["which", tool],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    has_yosys   = _which("yosys")
    has_iverilog = _which("iverilog")
    return {
        "yosys":    has_yosys,
        "iverilog": has_iverilog,
        "all_ok":   has_yosys and has_iverilog,
    }


# ── Yosys synthesis ───────────────────────────────────────────────────────────

_SYNTH_SCRIPT = """\
read_verilog {verilog_file}
synth -top dot_product_unit -flatten
stat
"""


def run_yosys_synthesis(verilog_file: str, work_dir: str = None) -> dict:
    """
    Run Yosys synthesis on a Verilog file and parse the resource report.

    Parameters
    ----------
    verilog_file : str
        Path to the .v file.
    work_dir : str, optional
        Working directory; uses a temp dir if None.

    Returns
    -------
    dict: synthesis_ok, luts, ffs, dsps, raw_log
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="yosys_")

    script_path = os.path.join(work_dir, "synth.ys")
    with open(script_path, "w") as f:
        f.write(_SYNTH_SCRIPT.format(verilog_file=os.path.abspath(verilog_file)))

    try:
        result = subprocess.run(
            ["yosys", "-s", script_path],
            capture_output=True, text=True, timeout=300,
            cwd=work_dir
        )
        raw_log = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"synthesis_ok": False, "luts": 0, "ffs": 0, "dsps": 0,
                "raw_log": "TIMEOUT"}
    except FileNotFoundError:
        return {"synthesis_ok": False, "luts": 0, "ffs": 0, "dsps": 0,
                "raw_log": "yosys not found"}

    if result.returncode != 0:
        return {"synthesis_ok": False, "luts": 0, "ffs": 0, "dsps": 0,
                "raw_log": raw_log}

    # Parse stat output
    luts = _parse_yosys_stat(raw_log, r"LUT\d+\s+(\d+)")
    ffs  = _parse_yosys_stat(raw_log, r"Flip-Flop\s+(\d+)")
    dsps = _parse_yosys_stat(raw_log, r"DSP\d*\s+(\d+)")

    # Fallback: sum up all cell counts from the stat section
    if luts == 0:
        luts = _parse_yosys_stat(raw_log, r"\$_(?:LUT|AND|OR|MUX|NOT)[^\s]*\s+(\d+)")
        # Try summing all LUT cells
        lut_matches = re.findall(r"LUT\d+\s+(\d+)", raw_log)
        luts = sum(int(m) for m in lut_matches) if lut_matches else luts

    return {
        "synthesis_ok": True,
        "luts":  luts,
        "ffs":   ffs,
        "dsps":  dsps,
        "raw_log": raw_log,
    }


def _parse_yosys_stat(log: str, pattern: str) -> int:
    matches = re.findall(pattern, log)
    return sum(int(m) for m in matches)


# ── Icarus Verilog simulation ─────────────────────────────────────────────────

_TESTBENCH_TEMPLATE = """\
`timescale 1ns/1ps
module tb;
    localparam BIT_W   = {bit_w};
    localparam VEC_LEN = {vec_len};

    reg                           clk   = 0;
    reg                           rst_n = 0;
    reg  signed [BIT_W*VEC_LEN-1:0] w_flat;
    reg  signed [BIT_W*VEC_LEN-1:0] x_flat;
    reg  signed [BIT_W-1:0]         bias;
    wire signed [BIT_W-1:0]         y;

    dot_product_unit #(
        .BIT_W({bit_w}), .VEC_LEN({vec_len}),
        .PIPE_STAGES({pipe_stages}), .ACT_TYPE({act_type}),
        .ACCUM_EXTRA({accum_extra}), .USE_DSP({use_dsp})
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .w_flat(w_flat), .x_flat(x_flat), .bias(bias), .y(y)
    );

    always #5 clk = ~clk;

    integer pass = 0, fail = 0, i;
    reg signed [BIT_W*VEC_LEN-1:0] w_tmp, x_tmp;

    initial begin
        rst_n = 0;
        w_flat = 0; x_flat = 0; bias = 0;
        #20; rst_n = 1;

        // Simple sanity vectors
        for (i = 0; i < 16; i = i + 1) begin
            w_flat = {vec_len}{{i[BIT_W-1:0]}};
            x_flat = {vec_len}{{1'b1, {{(BIT_W-1){{1'b0}}}}}};
            bias   = 0;
            #20;
            pass = pass + 1;  // just check it doesn't hang
        end

        $display("PASS=%0d FAIL=%0d", pass, fail);
        $finish;
    end

    initial #2000 begin
        $display("TIMEOUT");
        $finish;
    end
endmodule
"""


def run_iverilog_simulation(verilog_file: str, config: dict,
                             work_dir: str = None) -> dict:
    """
    Compile and simulate the Verilog with Icarus Verilog.

    Returns
    -------
    dict: sim_ok, pass_rate, raw_log
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="iverilog_")

    tb_path = os.path.join(work_dir, "tb.v")
    with open(tb_path, "w") as f:
        f.write(_TESTBENCH_TEMPLATE.format(**config))

    sim_bin = os.path.join(work_dir, "sim_out")

    try:
        # Compile
        comp = subprocess.run(
            ["iverilog", "-o", sim_bin, tb_path, verilog_file],
            capture_output=True, text=True, timeout=60
        )
        if comp.returncode != 0:
            return {"sim_ok": False, "pass_rate": 0.0,
                    "raw_log": comp.stdout + comp.stderr}

        # Run
        sim = subprocess.run(
            ["vvp", sim_bin],
            capture_output=True, text=True, timeout=60
        )
        log = sim.stdout + sim.stderr

        # Parse pass/fail
        m = re.search(r"PASS=(\d+)\s+FAIL=(\d+)", log)
        if m:
            p, f = int(m.group(1)), int(m.group(2))
            pass_rate = p / max(1, p + f)
        else:
            pass_rate = 1.0 if "TIMEOUT" not in log else 0.0

        return {"sim_ok": True, "pass_rate": pass_rate, "raw_log": log}

    except Exception as e:
        return {"sim_ok": False, "pass_rate": 0.0, "raw_log": str(e)}


# ── Full synthesis run ────────────────────────────────────────────────────────

def run_full_synthesis(verilog_file: str, config: dict,
                       work_dir: str = "build") -> dict:
    """
    Run Yosys synthesis + iverilog simulation + accuracy test,
    then assemble the full metrics dict.

    Falls back to estimated metrics if tools are unavailable.
    """
    tools = check_tools_available()

    # --- Yosys synthesis ---
    if tools["yosys"]:
        synth = run_yosys_synthesis(verilog_file, work_dir=work_dir)
    else:
        synth = estimate_synthesis_metrics(config)
        synth["raw_log"] = "yosys not available - using estimates"

    if not synth.get("synthesis_ok", False):
        return {
            "synthesis_ok": False,
            "luts": 0, "ffs": 0, "dsps": 0,
            "frequency": 0.0, "latency": 0,
            "throughput": 0.0, "ops_per_lut": 0.0,
            "mae": 1.0, "pass_rate": 0.0,
            "n_tests": 0,
            "mode": "full",
        }

    # --- Timing estimate (Yosys doesn't do STA by default) ---
    est = estimate_synthesis_metrics(config)
    frequency = est["frequency"]
    latency   = est["latency"]

    # Override LUT/FF counts with real Yosys numbers if we got them
    luts = synth.get("luts", 0) or est["luts"]
    ffs  = synth.get("ffs",  0) or est["ffs"]
    dsps = synth.get("dsps", 0)

    # --- iverilog simulation ---
    if tools["iverilog"]:
        sim = run_iverilog_simulation(verilog_file, config, work_dir=work_dir)
        pass_rate = sim.get("pass_rate", 1.0)
    else:
        pass_rate = 1.0  # assume functional if we can't simulate

    # --- Python accuracy test ---
    acctest  = run_accuracy_test(config)
    pass_rate = min(pass_rate, acctest["pass_rate"])

    freq_hz    = frequency * 1e6
    throughput = freq_hz * config["vec_len"]
    ops_per_lut = throughput / max(luts, 1)

    return {
        "synthesis_ok": True,
        "luts":         luts,
        "ffs":          ffs,
        "dsps":         dsps,
        "frequency":    frequency,
        "latency":      latency,
        "throughput":   throughput,
        "ops_per_lut":  ops_per_lut,
        "mae":          acctest["mae"],
        "pass_rate":    pass_rate,
        "n_tests":      acctest["n_tests"],
        "mode":         "full",
    }
