module rotmix (
  input  wire [63:0] x,
  output wire [63:0] y
);
  wire [63:0] r1 = {x[60:0], x[63:61]};
  wire [63:0] r2 = {x[12:0], x[63:13]};
  assign y = r1 ^ r2 ^ (x + 64'h9e3779b97f4a7c15);
endmodule
