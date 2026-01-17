module top(
  input logic clk,
  input logic rst,
  input logic [7:0] x,
  input logic [7:0] y,
  output logic [1:0] m
);
  assign m[0] = x[3:0] < y[3:0];
  assign m[1] = x[7:4] < y[7:4];
endmodule

