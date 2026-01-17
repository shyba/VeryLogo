module top(
  input logic clk,
  input logic rst,
  input logic [127:0] g,
  input logic [127:0] h,
  output logic [127:0] f
);
  wire [15:0] g0 = g[15:0];
  wire [15:0] g1 = g[31:16];
  wire [15:0] g2 = g[47:32];
  wire [15:0] g3 = g[63:48];
  wire [15:0] g4 = g[79:64];
  wire [15:0] g5 = g[95:80];
  wire [15:0] g6 = g[111:96];
  wire [15:0] g7 = g[127:112];

  wire [15:0] h0 = h[15:0];
  wire [15:0] h1 = h[31:16];
  wire [15:0] h2 = h[47:32];
  wire [15:0] h3 = h[63:48];
  wire [15:0] h4 = h[79:64];
  wire [15:0] h5 = h[95:80];
  wire [15:0] h6 = h[111:96];
  wire [15:0] h7 = h[127:112];

  wire [15:0] f0 = g0 + h0;
  wire [15:0] f1 = g1 + h1;
  wire [15:0] f2 = g2 + h2;
  wire [15:0] f3 = g3 + h3;
  wire [15:0] f4 = g4 + h4;
  wire [15:0] f5 = g5 + h5;
  wire [15:0] f6 = g6 + h6;
  wire [15:0] f7 = g7 + h7;

  assign f = { f7, f6, f5, f4, f3, f2, f1, f0 };
endmodule

