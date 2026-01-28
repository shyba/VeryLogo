module nonlinear_island (
  input  wire [63:0] x,
  input  wire [63:0] y,
  input  wire [63:0] z,
  output wire [63:0] out
);
  wire [15:0] core0 = x[15:0] & y[15:0];
  wire [15:0] core1 = (x[31:16] & y[31:16]) ^ (x[31:16] & z[31:16]);
  wire [15:0] core2 = (y[47:32] & z[47:32]) ^ (x[47:32] & z[47:32]);
  wire [15:0] core3 = x[63:48] & y[63:48] & z[63:48];

  wire [63:0] nl_out = {core3, core2, core1, core0};

  wire [63:0] mix1 = nl_out ^ {nl_out[50:0], nl_out[63:51]};
  wire [63:0] mix2 = mix1 ^ {mix1[28:0], mix1[63:29]};
  wire [63:0] mix3 = mix2 ^ {mix2[39:0], mix2[63:40]};
  wire [63:0] mix4 = mix3 ^ {mix3[17:0], mix3[63:18]};
  wire [63:0] mix5 = mix4 ^ {mix4[44:0], mix4[63:44]};
  wire [63:0] mix6 = mix5 ^ {mix5[23:0], mix5[63:23]};

  assign out = mix6 ^ nl_out;
endmodule
