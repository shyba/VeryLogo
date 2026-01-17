module top(input wire clk, input wire rst, input wire i, output wire o);
  assign o = ~i;
endmodule

