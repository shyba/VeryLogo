module slice_concat_patho (
  input  wire [63:0] x,
  output wire [63:0] y
);
  wire [31:0] hi = x[63:32];
  wire [31:0] lo = x[31:0];

  wire [15:0] q0 = lo[15:0];
  wire [15:0] q1 = lo[31:16];
  wire [15:0] q2 = hi[15:0];
  wire [15:0] q3 = hi[31:16];

  wire [63:0] permuted = {q1, q3, q0, q2};

  wire [23:0] slice_a = permuted[47:24];
  wire [23:0] slice_b = {permuted[23:0]};
  wire [15:0] slice_c = permuted[63:48];

  wire [63:0] recon = {slice_c, slice_a, slice_b};

  wire [9:0] odd_slice_0 = recon[9:0];
  wire [9:0] odd_slice_1 = recon[22:13];
  wire [9:0] odd_slice_2 = recon[35:26];
  wire [9:0] odd_slice_3 = recon[48:39];
  wire [23:0] odd_slice_4 = {recon[63:49], recon[12:10], recon[25:23], recon[38:36]};

  assign y = {odd_slice_3, odd_slice_4, odd_slice_2, odd_slice_1, odd_slice_0};
endmodule
