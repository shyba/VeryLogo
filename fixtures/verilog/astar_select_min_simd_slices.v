module top(
  input logic clk,
  input logic [127:0] a,
  input logic [127:0] b,
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

  wire [15:0] o0 = (a0 < b0) ? a0 : b0;
  wire [15:0] o1 = (a1 < b1) ? a1 : b1;
  wire [15:0] o2 = (a2 < b2) ? a2 : b2;
  wire [15:0] o3 = (a3 < b3) ? a3 : b3;
  wire [15:0] o4 = (a4 < b4) ? a4 : b4;
  wire [15:0] o5 = (a5 < b5) ? a5 : b5;
  wire [15:0] o6 = (a6 < b6) ? a6 : b6;
  wire [15:0] o7 = (a7 < b7) ? a7 : b7;

  assign o = { o7, o6, o5, o4, o3, o2, o1, o0 };
endmodule
