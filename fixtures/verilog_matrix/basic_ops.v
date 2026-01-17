module top(
  input logic clk,
  input logic rst,
  input logic sel,
  input logic [7:0] a,
  input logic [7:0] b,
  output logic [7:0] o_not,
  output logic [7:0] o_and,
  output logic [7:0] o_or,
  output logic [7:0] o_xor,
  output logic [7:0] o_add,
  output logic [7:0] o_sub,
  output logic [7:0] o_mux,
  output logic o_eq
);
  assign o_not = ~a;
  assign o_and = a & b;
  assign o_or = a | b;
  assign o_xor = a ^ b;
  assign o_add = a + b;
  assign o_sub = a - b;
  assign o_mux = sel ? a : b;
  assign o_eq = (a == b);
endmodule

