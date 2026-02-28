"""
verilog_gen.py - Synthesizable Verilog generator for the dot-product unit.
Produces real, parameterized RTL that Yosys can synthesize.
"""

import os
import math


def generate_verilog(config: dict, iteration: int = 0,
                     build_dir: str = "build") -> str:
    """
    Generate a synthesizable dot_product_unit Verilog file.

    Parameters
    ----------
    config : dict
        Keys: bit_w, vec_len, pipe_stages, act_type, accum_extra, use_dsp
    iteration : int
        Loop iteration number (embedded in file header comment).
    build_dir : str
        Directory to write the .v file.

    Returns
    -------
    str
        Absolute path to the generated file.
    """
    bit_w       = config["bit_w"]
    vec_len     = config["vec_len"]
    pipe_stages = config["pipe_stages"]
    act_type    = config["act_type"]
    accum_extra = config["accum_extra"]
    use_dsp     = config["use_dsp"]

    accum_w = bit_w * 2 + accum_extra + int(math.ceil(math.log2(vec_len)))

    act_str = {0: "none", 1: "relu", 2: "clamp"}[act_type]
    dsp_str = "DSP" if use_dsp else "LUT"

    # Build genvar-based parallel multiplier block
    mult_lines = []
    mult_lines.append(f"    // Parallel multiplier array ({vec_len} units, {dsp_str})")
    mult_lines.append(f"    genvar i;")
    mult_lines.append(f"    generate")
    mult_lines.append(f"        for (i = 0; i < VEC_LEN; i = i + 1) begin : mult_array")
    if use_dsp:
        mult_lines.append(f"            (* use_dsp = \"yes\" *)")
    else:
        mult_lines.append(f"            (* use_dsp = \"no\" *)")
    mult_lines.append(f"            assign products[i] = $signed(w[i]) * $signed(x[i]);")
    mult_lines.append(f"        end")
    mult_lines.append(f"    endgenerate")

    # Adder tree (log2 levels)
    adder_levels = int(math.ceil(math.log2(max(vec_len, 2))))
    adder_lines = []
    adder_lines.append(f"    // {adder_levels}-level adder tree with sign extension")

    # Build partial sums
    adder_lines.append(f"    wire signed [{accum_w-1}:0] psum [0:{vec_len-1}];")
    adder_lines.append(f"    genvar j;")
    adder_lines.append(f"    generate")
    adder_lines.append(f"        for (j = 0; j < VEC_LEN; j = j + 1) begin : psum_init")
    adder_lines.append(f"            assign psum[j] = {{{{({accum_w}-PROD_W){{products[j][PROD_W-1]}}}}, products[j]}};")
    adder_lines.append(f"        end")
    adder_lines.append(f"    endgenerate")

    adder_lines.append(f"")
    adder_lines.append(f"    // Sum all partial sums")
    adder_lines.append(f"    wire signed [{accum_w-1}:0] tree_sum;")

    # Build the summation expression
    sum_parts = [f"psum[{k}]" for k in range(vec_len)]
    sum_expr = " + ".join(sum_parts)
    adder_lines.append(f"    assign tree_sum = {sum_expr};")

    # Bias addition
    adder_lines.append(f"    wire signed [{accum_w-1}:0] biased_sum;")
    adder_lines.append(f"    assign biased_sum = tree_sum + $signed({{{{({accum_w}-BIT_W){{bias[BIT_W-1]}}}}, bias}});")

    # Activation function
    act_lines = []
    act_lines.append(f"    // Activation function: {act_str}")
    act_lines.append(f"    wire signed [{bit_w-1}:0] act_out;")
    clamp_max = (1 << (bit_w - 1)) - 1
    clamp_min = -(1 << (bit_w - 1))

    act_lines.append(f"    wire signed [{accum_w-1}:0] act_in;")
    act_lines.append(f"    assign act_in = biased_sum;")
    act_lines.append(f"    assign act_out =")

    if act_type == 0:  # none
        act_lines.append(f"        act_in[{bit_w-1}:0];  // truncate to output width")
    elif act_type == 1:  # ReLU
        act_lines.append(f"        (act_in[{accum_w-1}]) ? {bit_w}'d0 :  // negative -> 0")
        act_lines.append(f"        (act_in > {accum_w}'sd{clamp_max}) ? {bit_w}'sd{clamp_max} :")
        act_lines.append(f"        act_in[{bit_w-1}:0];")
    else:  # clamp
        act_lines.append(f"        (act_in > {accum_w}'sd{clamp_max}) ? {bit_w}'sd{clamp_max} :")
        act_lines.append(f"        (act_in < {accum_w}'sd{clamp_min}) ? {bit_w}'sd{clamp_min} :")
        act_lines.append(f"        act_in[{bit_w-1}:0];")

    # Pipeline registers
    pipe_lines = []
    pipe_lines.append(f"    // Pipeline registers (depth={pipe_stages})")
    if pipe_stages == 1:
        pipe_lines.append(f"    reg signed [{bit_w-1}:0] y_reg;")
        pipe_lines.append(f"    always @(posedge clk or negedge rst_n) begin")
        pipe_lines.append(f"        if (!rst_n) y_reg <= {bit_w}'d0;")
        pipe_lines.append(f"        else        y_reg <= act_out;")
        pipe_lines.append(f"    end")
    elif pipe_stages == 2:
        pipe_lines.append(f"    reg signed [{bit_w-1}:0] y_p1, y_reg;")
        pipe_lines.append(f"    always @(posedge clk or negedge rst_n) begin")
        pipe_lines.append(f"        if (!rst_n) begin y_p1 <= {bit_w}'d0; y_reg <= {bit_w}'d0; end")
        pipe_lines.append(f"        else        begin y_p1 <= act_out;   y_reg <= y_p1;   end")
        pipe_lines.append(f"    end")
    else:  # 3 stages
        pipe_lines.append(f"    reg signed [{bit_w-1}:0] y_p1, y_p2, y_reg;")
        pipe_lines.append(f"    always @(posedge clk or negedge rst_n) begin")
        pipe_lines.append(f"        if (!rst_n) begin y_p1<={bit_w}'d0; y_p2<={bit_w}'d0; y_reg<={bit_w}'d0; end")
        pipe_lines.append(f"        else        begin y_p1<=act_out; y_p2<=y_p1; y_reg<=y_p2; end")
        pipe_lines.append(f"    end")

    verilog = f"""\
// =============================================================================
// dot_product_unit.v
// AI-FPGA-Loop: Auto-generated synthesizable RTL
// Iteration   : {iteration}
// Config      : bit_w={bit_w} vec_len={vec_len} pipe={pipe_stages}
//               act={act_str} accum_extra={accum_extra} mult={dsp_str}
// =============================================================================
`timescale 1ns/1ps

module dot_product_unit #(
    parameter BIT_W       = {bit_w},   // data width (bits)
    parameter VEC_LEN     = {vec_len},   // dot-product vector length
    parameter PIPE_STAGES = {pipe_stages},   // pipeline depth
    parameter ACT_TYPE    = {act_type},   // 0=none, 1=relu, 2=clamp
    parameter ACCUM_EXTRA = {accum_extra},   // extra accumulator bits
    parameter USE_DSP     = {use_dsp},   // 0=LUT mult, 1=DSP mult
    parameter ACCUM_W     = {accum_w},   // accumulator width (auto)
    parameter PROD_W      = BIT_W * 2   // product width
) (
    input  wire                           clk,
    input  wire                           rst_n,   // async active-low reset
    input  wire signed [BIT_W*VEC_LEN-1:0] w_flat, // weight vector (packed)
    input  wire signed [BIT_W*VEC_LEN-1:0] x_flat, // input  vector (packed)
    input  wire signed [BIT_W-1:0]         bias,
    output reg  signed [BIT_W-1:0]         y       // output (pipelined)
);

    // Unpack flat ports into arrays
    wire signed [BIT_W-1:0] w [0:VEC_LEN-1];
    wire signed [BIT_W-1:0] x [0:VEC_LEN-1];
    genvar k;
    generate
        for (k = 0; k < VEC_LEN; k = k + 1) begin : unpack
            assign w[k] = w_flat[BIT_W*(k+1)-1 : BIT_W*k];
            assign x[k] = x_flat[BIT_W*(k+1)-1 : BIT_W*k];
        end
    endgenerate

    // Product wires
    wire signed [PROD_W-1:0] products [0:VEC_LEN-1];

{chr(10).join(mult_lines)}

{chr(10).join(adder_lines)}

{chr(10).join(act_lines)}

{chr(10).join(pipe_lines)}

    // Output assignment
    always @(*) y = y_reg;

endmodule
// end of dot_product_unit
"""

    os.makedirs(build_dir, exist_ok=True)
    fname = os.path.join(build_dir, f"iter_{iteration:04d}_dot_product_unit.v")
    with open(fname, "w") as f:
        f.write(verilog)

    return fname


def save_best_verilog(config: dict, build_dir: str = "build") -> str:
    """Save the best design as BEST_DESIGN.v."""
    bit_w       = config["bit_w"]
    vec_len     = config["vec_len"]
    pipe_stages = config["pipe_stages"]
    act_type    = config["act_type"]
    accum_extra = config["accum_extra"]
    use_dsp     = config["use_dsp"]

    accum_w = bit_w * 2 + accum_extra + int(math.ceil(math.log2(vec_len)))
    act_str = {0: "none", 1: "relu", 2: "clamp"}[act_type]
    dsp_str = "DSP" if use_dsp else "LUT"

    mult_lines = []
    mult_lines.append(f"    genvar i;")
    mult_lines.append(f"    generate")
    mult_lines.append(f"        for (i = 0; i < VEC_LEN; i = i + 1) begin : mult_array")
    if use_dsp:
        mult_lines.append(f"            (* use_dsp = \"yes\" *)")
    else:
        mult_lines.append(f"            (* use_dsp = \"no\" *)")
    mult_lines.append(f"            assign products[i] = $signed(w[i]) * $signed(x[i]);")
    mult_lines.append(f"        end")
    mult_lines.append(f"    endgenerate")

    adder_lines = []
    adder_lines.append(f"    wire signed [{accum_w-1}:0] psum [0:{vec_len-1}];")
    adder_lines.append(f"    genvar j;")
    adder_lines.append(f"    generate")
    adder_lines.append(f"        for (j = 0; j < VEC_LEN; j = j + 1) begin : psum_init")
    adder_lines.append(f"            assign psum[j] = {{{{({accum_w}-PROD_W){{products[j][PROD_W-1]}}}}, products[j]}};")
    adder_lines.append(f"        end")
    adder_lines.append(f"    endgenerate")
    adder_lines.append(f"    wire signed [{accum_w-1}:0] tree_sum;")
    sum_parts = [f"psum[{k}]" for k in range(vec_len)]
    adder_lines.append(f"    assign tree_sum = {' + '.join(sum_parts)};")
    adder_lines.append(f"    wire signed [{accum_w-1}:0] biased_sum;")
    adder_lines.append(f"    assign biased_sum = tree_sum + $signed({{{{({accum_w}-BIT_W){{bias[BIT_W-1]}}}}, bias}});")

    clamp_max = (1 << (bit_w - 1)) - 1
    clamp_min = -(1 << (bit_w - 1))
    act_lines = []
    act_lines.append(f"    wire signed [{bit_w-1}:0] act_out;")
    act_lines.append(f"    wire signed [{accum_w-1}:0] act_in;")
    act_lines.append(f"    assign act_in = biased_sum;")
    act_lines.append(f"    assign act_out =")
    if act_type == 0:
        act_lines.append(f"        act_in[{bit_w-1}:0];")
    elif act_type == 1:
        act_lines.append(f"        (act_in[{accum_w-1}]) ? {bit_w}'d0 :")
        act_lines.append(f"        (act_in > {accum_w}'sd{clamp_max}) ? {bit_w}'sd{clamp_max} :")
        act_lines.append(f"        act_in[{bit_w-1}:0];")
    else:
        act_lines.append(f"        (act_in > {accum_w}'sd{clamp_max}) ? {bit_w}'sd{clamp_max} :")
        act_lines.append(f"        (act_in < {accum_w}'sd{clamp_min}) ? {bit_w}'sd{clamp_min} :")
        act_lines.append(f"        act_in[{bit_w-1}:0];")

    pipe_lines = []
    if pipe_stages == 1:
        pipe_lines += [
            f"    reg signed [{bit_w-1}:0] y_reg;",
            f"    always @(posedge clk or negedge rst_n) begin",
            f"        if (!rst_n) y_reg <= {bit_w}'d0;",
            f"        else        y_reg <= act_out;",
            f"    end",
        ]
    elif pipe_stages == 2:
        pipe_lines += [
            f"    reg signed [{bit_w-1}:0] y_p1, y_reg;",
            f"    always @(posedge clk or negedge rst_n) begin",
            f"        if (!rst_n) begin y_p1 <= {bit_w}'d0; y_reg <= {bit_w}'d0; end",
            f"        else        begin y_p1 <= act_out;   y_reg <= y_p1;   end",
            f"    end",
        ]
    else:
        pipe_lines += [
            f"    reg signed [{bit_w-1}:0] y_p1, y_p2, y_reg;",
            f"    always @(posedge clk or negedge rst_n) begin",
            f"        if (!rst_n) begin y_p1<={bit_w}'d0; y_p2<={bit_w}'d0; y_reg<={bit_w}'d0; end",
            f"        else        begin y_p1<=act_out; y_p2<=y_p1; y_reg<=y_p2; end",
            f"    end",
        ]

    verilog = f"""\
// =============================================================================
// BEST_DESIGN.v  -  AI-FPGA-Loop Best Design
// Config: bit_w={bit_w} vec_len={vec_len} pipe_stages={pipe_stages}
//         act={act_str} accum_extra={accum_extra} mult={dsp_str}
// =============================================================================
`timescale 1ns/1ps

module dot_product_unit #(
    parameter BIT_W       = {bit_w},
    parameter VEC_LEN     = {vec_len},
    parameter PIPE_STAGES = {pipe_stages},
    parameter ACT_TYPE    = {act_type},
    parameter ACCUM_EXTRA = {accum_extra},
    parameter USE_DSP     = {use_dsp},
    parameter ACCUM_W     = {accum_w},
    parameter PROD_W      = BIT_W * 2
) (
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire signed [BIT_W*VEC_LEN-1:0] w_flat,
    input  wire signed [BIT_W*VEC_LEN-1:0] x_flat,
    input  wire signed [BIT_W-1:0]         bias,
    output reg  signed [BIT_W-1:0]         y
);

    wire signed [BIT_W-1:0] w [0:VEC_LEN-1];
    wire signed [BIT_W-1:0] x [0:VEC_LEN-1];
    genvar k;
    generate
        for (k = 0; k < VEC_LEN; k = k + 1) begin : unpack
            assign w[k] = w_flat[BIT_W*(k+1)-1 : BIT_W*k];
            assign x[k] = x_flat[BIT_W*(k+1)-1 : BIT_W*k];
        end
    endgenerate

    wire signed [PROD_W-1:0] products [0:VEC_LEN-1];

{chr(10).join(mult_lines)}

{chr(10).join(adder_lines)}

{chr(10).join(act_lines)}

{chr(10).join(pipe_lines)}

    always @(*) y = y_reg;

endmodule
"""

    os.makedirs(build_dir, exist_ok=True)
    best_path = os.path.join(build_dir, "BEST_DESIGN.v")
    with open(best_path, "w") as f:
        f.write(verilog)
    return best_path
