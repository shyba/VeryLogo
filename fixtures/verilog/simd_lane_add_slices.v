module top(
  input logic clk,
  input logic rst,
  input logic [7:0] x,
  input logic [7:0] y,
  output logic [7:0] o
);
  assign o = { (x[7:4] + y[7:4]), (x[3:0] + y[3:0]) };
endmodule

