module top(
  input logic clk,
  input logic rst,
  input logic [127:0] a,
  input logic [127:0] b,
  input logic [127:0] x,
  input logic [127:0] y,
  output logic [127:0] o
);
  wire [15:0] a0 = a[15:0];
  wire [15:0] a1 = a[31:16];
  wire [15:0] a2 = a[47:32];
  wire [15:0] a3 = a[63:48];
  wire [15:0] a4 = a[79:64];
  wire [15:0] a5 = a[95:80];
  wire [15:0] a6 = a[111:96];
  wire [15:0] a7 = a[127:112];

  wire [15:0] b0 = b[15:0];
  wire [15:0] b1 = b[31:16];
  wire [15:0] b2 = b[47:32];
  wire [15:0] b3 = b[63:48];
  wire [15:0] b4 = b[79:64];
  wire [15:0] b5 = b[95:80];
  wire [15:0] b6 = b[111:96];
  wire [15:0] b7 = b[127:112];

  wire [15:0] x0 = x[15:0];
  wire [15:0] x1 = x[31:16];
  wire [15:0] x2 = x[47:32];
  wire [15:0] x3 = x[63:48];
  wire [15:0] x4 = x[79:64];
  wire [15:0] x5 = x[95:80];
  wire [15:0] x6 = x[111:96];
  wire [15:0] x7 = x[127:112];

  wire [15:0] y0 = y[15:0];
  wire [15:0] y1 = y[31:16];
  wire [15:0] y2 = y[47:32];
  wire [15:0] y3 = y[63:48];
  wire [15:0] y4 = y[79:64];
  wire [15:0] y5 = y[95:80];
  wire [15:0] y6 = y[111:96];
  wire [15:0] y7 = y[127:112];

  wire [15:0] o0 = (a0 == b0) ? x0 : y0;
  wire [15:0] o1 = (a1 == b1) ? x1 : y1;
  wire [15:0] o2 = (a2 == b2) ? x2 : y2;
  wire [15:0] o3 = (a3 == b3) ? x3 : y3;
  wire [15:0] o4 = (a4 == b4) ? x4 : y4;
  wire [15:0] o5 = (a5 == b5) ? x5 : y5;
  wire [15:0] o6 = (a6 == b6) ? x6 : y6;
  wire [15:0] o7 = (a7 == b7) ? x7 : y7;

  assign o = { o7, o6, o5, o4, o3, o2, o1, o0 };
endmodule

