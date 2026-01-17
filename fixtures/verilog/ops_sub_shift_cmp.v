module top(
  input logic clk,
  input logic rst,
  input logic [3:0] a,
  input logic [3:0] b,
  input logic [1:0] sh,
  output logic [3:0] o_sub,
  output logic [3:0] o_shl,
  output logic [3:0] o_lshr,
  output logic [3:0] o_ashr,
  output logic lt,
  output logic le,
  output logic gt,
  output logic ge
);
  assign o_sub = a - b;
  assign o_shl = a << sh;
  assign o_lshr = a >> sh;
  assign o_ashr = $signed(a) >>> sh;
  assign lt = (a < b);
  assign le = (a <= b);
  assign gt = (a > b);
  assign ge = (a >= b);
endmodule

