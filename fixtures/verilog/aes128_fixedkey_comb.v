module aes128_fixedkey_comb(
  input wire [127:0] pt,
  output wire [127:0] ct
);
  reg [7:0] sbox [0:255];
  initial $readmemh("fixtures/verilog/aes_sbox.mem", sbox);

  function automatic [7:0] xtime(input [7:0] x);
    begin
      xtime = {x[6:0], 1'b0} ^ (8'h1b & {8{x[7]}});
    end
  endfunction

  function automatic [31:0] mixcol(input [31:0] c);
    reg [7:0] a0;
    reg [7:0] a1;
    reg [7:0] a2;
    reg [7:0] a3;
    reg [7:0] b0;
    reg [7:0] b1;
    reg [7:0] b2;
    reg [7:0] b3;
    reg [7:0] m2_0;
    reg [7:0] m2_1;
    reg [7:0] m2_2;
    reg [7:0] m2_3;
    reg [7:0] m3_0;
    reg [7:0] m3_1;
    reg [7:0] m3_2;
    reg [7:0] m3_3;
    begin
      a0 = c[31:24];
      a1 = c[23:16];
      a2 = c[15:8];
      a3 = c[7:0];
      m2_0 = xtime(a0);
      m2_1 = xtime(a1);
      m2_2 = xtime(a2);
      m2_3 = xtime(a3);
      m3_0 = m2_0 ^ a0;
      m3_1 = m2_1 ^ a1;
      m3_2 = m2_2 ^ a2;
      m3_3 = m2_3 ^ a3;
      b0 = m2_0 ^ m3_1 ^ a2 ^ a3;
      b1 = a0 ^ m2_1 ^ m3_2 ^ a3;
      b2 = a0 ^ a1 ^ m2_2 ^ m3_3;
      b3 = m3_0 ^ a1 ^ a2 ^ m2_3;
      mixcol = {b0, b1, b2, b3};
    end
  endfunction

  localparam [127:0] RK0 = 128'h000102030405060708090a0b0c0d0e0f;
  localparam [127:0] RK1 = 128'hd6aa74fdd2af72fadaa678f1d6ab76fe;
  localparam [127:0] RK2 = 128'hb692cf0b643dbdf1be9bc5006830b3fe;
  localparam [127:0] RK3 = 128'hb6ff744ed2c2c9bf6c590cbf0469bf41;
  localparam [127:0] RK4 = 128'h47f7f7bc95353e03f96c32bcfd058dfd;
  localparam [127:0] RK5 = 128'h3caaa3e8a99f9deb50f3af57adf622aa;
  localparam [127:0] RK6 = 128'h5e390f7df7a69296a7553dc10aa31f6b;
  localparam [127:0] RK7 = 128'h14f9701ae35fe28c440adf4d4ea9c026;
  localparam [127:0] RK8 = 128'h47438735a41c65b9e016baf4aebf7ad2;
  localparam [127:0] RK9 = 128'h549932d1f08557681093ed9cbe2c974e;
  localparam [127:0] RK10 = 128'h13111d7fe3944a17f307a78b4d2b30c5;

  wire [7:0] st0 [0:15];
  wire [7:0] st1 [0:15];
  wire [7:0] st2 [0:15];
  wire [7:0] st3 [0:15];
  wire [7:0] st4 [0:15];
  wire [7:0] st5 [0:15];
  wire [7:0] st6 [0:15];
  wire [7:0] st7 [0:15];
  wire [7:0] st8 [0:15];
  wire [7:0] st9 [0:15];
  wire [7:0] st10 [0:15];
  wire [7:0] sb1 [0:15];
  wire [7:0] sr1 [0:15];
  wire [7:0] mc1 [0:15];
  wire [7:0] sb2 [0:15];
  wire [7:0] sr2 [0:15];
  wire [7:0] mc2 [0:15];
  wire [7:0] sb3 [0:15];
  wire [7:0] sr3 [0:15];
  wire [7:0] mc3 [0:15];
  wire [7:0] sb4 [0:15];
  wire [7:0] sr4 [0:15];
  wire [7:0] mc4 [0:15];
  wire [7:0] sb5 [0:15];
  wire [7:0] sr5 [0:15];
  wire [7:0] mc5 [0:15];
  wire [7:0] sb6 [0:15];
  wire [7:0] sr6 [0:15];
  wire [7:0] mc6 [0:15];
  wire [7:0] sb7 [0:15];
  wire [7:0] sr7 [0:15];
  wire [7:0] mc7 [0:15];
  wire [7:0] sb8 [0:15];
  wire [7:0] sr8 [0:15];
  wire [7:0] mc8 [0:15];
  wire [7:0] sb9 [0:15];
  wire [7:0] sr9 [0:15];
  wire [7:0] mc9 [0:15];
  wire [7:0] sb10 [0:15];
  wire [7:0] sr10 [0:15];

  genvar i;
  generate
    for (i = 0; i < 16; i = i + 1) begin : gen_rk0
      assign st0[i] = pt[127 - (i * 8) -: 8] ^ RK0[127 - (i * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb1
      assign sb1[j] = sbox[st0[j]];
    end
  endgenerate

  assign sr1[0] = sb1[0];
  assign sr1[1] = sb1[5];
  assign sr1[2] = sb1[10];
  assign sr1[3] = sb1[15];
  assign sr1[4] = sb1[4];
  assign sr1[5] = sb1[9];
  assign sr1[6] = sb1[14];
  assign sr1[7] = sb1[3];
  assign sr1[8] = sb1[8];
  assign sr1[9] = sb1[13];
  assign sr1[10] = sb1[2];
  assign sr1[11] = sb1[7];
  assign sr1[12] = sb1[12];
  assign sr1[13] = sb1[1];
  assign sr1[14] = sb1[6];
  assign sr1[15] = sb1[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc1
      wire [31:0] col = mixcol({sr1[c * 4 + 0], sr1[c * 4 + 1], sr1[c * 4 + 2], sr1[c * 4 + 3]});
      assign mc1[c * 4 + 0] = col[31:24];
      assign mc1[c * 4 + 1] = col[23:16];
      assign mc1[c * 4 + 2] = col[15:8];
      assign mc1[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add1
      assign st1[k] = mc1[k] ^ RK1[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb2
      assign sb2[j] = sbox[st1[j]];
    end
  endgenerate

  assign sr2[0] = sb2[0];
  assign sr2[1] = sb2[5];
  assign sr2[2] = sb2[10];
  assign sr2[3] = sb2[15];
  assign sr2[4] = sb2[4];
  assign sr2[5] = sb2[9];
  assign sr2[6] = sb2[14];
  assign sr2[7] = sb2[3];
  assign sr2[8] = sb2[8];
  assign sr2[9] = sb2[13];
  assign sr2[10] = sb2[2];
  assign sr2[11] = sb2[7];
  assign sr2[12] = sb2[12];
  assign sr2[13] = sb2[1];
  assign sr2[14] = sb2[6];
  assign sr2[15] = sb2[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc2
      wire [31:0] col = mixcol({sr2[c * 4 + 0], sr2[c * 4 + 1], sr2[c * 4 + 2], sr2[c * 4 + 3]});
      assign mc2[c * 4 + 0] = col[31:24];
      assign mc2[c * 4 + 1] = col[23:16];
      assign mc2[c * 4 + 2] = col[15:8];
      assign mc2[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add2
      assign st2[k] = mc2[k] ^ RK2[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb3
      assign sb3[j] = sbox[st2[j]];
    end
  endgenerate

  assign sr3[0] = sb3[0];
  assign sr3[1] = sb3[5];
  assign sr3[2] = sb3[10];
  assign sr3[3] = sb3[15];
  assign sr3[4] = sb3[4];
  assign sr3[5] = sb3[9];
  assign sr3[6] = sb3[14];
  assign sr3[7] = sb3[3];
  assign sr3[8] = sb3[8];
  assign sr3[9] = sb3[13];
  assign sr3[10] = sb3[2];
  assign sr3[11] = sb3[7];
  assign sr3[12] = sb3[12];
  assign sr3[13] = sb3[1];
  assign sr3[14] = sb3[6];
  assign sr3[15] = sb3[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc3
      wire [31:0] col = mixcol({sr3[c * 4 + 0], sr3[c * 4 + 1], sr3[c * 4 + 2], sr3[c * 4 + 3]});
      assign mc3[c * 4 + 0] = col[31:24];
      assign mc3[c * 4 + 1] = col[23:16];
      assign mc3[c * 4 + 2] = col[15:8];
      assign mc3[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add3
      assign st3[k] = mc3[k] ^ RK3[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb4
      assign sb4[j] = sbox[st3[j]];
    end
  endgenerate

  assign sr4[0] = sb4[0];
  assign sr4[1] = sb4[5];
  assign sr4[2] = sb4[10];
  assign sr4[3] = sb4[15];
  assign sr4[4] = sb4[4];
  assign sr4[5] = sb4[9];
  assign sr4[6] = sb4[14];
  assign sr4[7] = sb4[3];
  assign sr4[8] = sb4[8];
  assign sr4[9] = sb4[13];
  assign sr4[10] = sb4[2];
  assign sr4[11] = sb4[7];
  assign sr4[12] = sb4[12];
  assign sr4[13] = sb4[1];
  assign sr4[14] = sb4[6];
  assign sr4[15] = sb4[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc4
      wire [31:0] col = mixcol({sr4[c * 4 + 0], sr4[c * 4 + 1], sr4[c * 4 + 2], sr4[c * 4 + 3]});
      assign mc4[c * 4 + 0] = col[31:24];
      assign mc4[c * 4 + 1] = col[23:16];
      assign mc4[c * 4 + 2] = col[15:8];
      assign mc4[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add4
      assign st4[k] = mc4[k] ^ RK4[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb5
      assign sb5[j] = sbox[st4[j]];
    end
  endgenerate

  assign sr5[0] = sb5[0];
  assign sr5[1] = sb5[5];
  assign sr5[2] = sb5[10];
  assign sr5[3] = sb5[15];
  assign sr5[4] = sb5[4];
  assign sr5[5] = sb5[9];
  assign sr5[6] = sb5[14];
  assign sr5[7] = sb5[3];
  assign sr5[8] = sb5[8];
  assign sr5[9] = sb5[13];
  assign sr5[10] = sb5[2];
  assign sr5[11] = sb5[7];
  assign sr5[12] = sb5[12];
  assign sr5[13] = sb5[1];
  assign sr5[14] = sb5[6];
  assign sr5[15] = sb5[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc5
      wire [31:0] col = mixcol({sr5[c * 4 + 0], sr5[c * 4 + 1], sr5[c * 4 + 2], sr5[c * 4 + 3]});
      assign mc5[c * 4 + 0] = col[31:24];
      assign mc5[c * 4 + 1] = col[23:16];
      assign mc5[c * 4 + 2] = col[15:8];
      assign mc5[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add5
      assign st5[k] = mc5[k] ^ RK5[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb6
      assign sb6[j] = sbox[st5[j]];
    end
  endgenerate

  assign sr6[0] = sb6[0];
  assign sr6[1] = sb6[5];
  assign sr6[2] = sb6[10];
  assign sr6[3] = sb6[15];
  assign sr6[4] = sb6[4];
  assign sr6[5] = sb6[9];
  assign sr6[6] = sb6[14];
  assign sr6[7] = sb6[3];
  assign sr6[8] = sb6[8];
  assign sr6[9] = sb6[13];
  assign sr6[10] = sb6[2];
  assign sr6[11] = sb6[7];
  assign sr6[12] = sb6[12];
  assign sr6[13] = sb6[1];
  assign sr6[14] = sb6[6];
  assign sr6[15] = sb6[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc6
      wire [31:0] col = mixcol({sr6[c * 4 + 0], sr6[c * 4 + 1], sr6[c * 4 + 2], sr6[c * 4 + 3]});
      assign mc6[c * 4 + 0] = col[31:24];
      assign mc6[c * 4 + 1] = col[23:16];
      assign mc6[c * 4 + 2] = col[15:8];
      assign mc6[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add6
      assign st6[k] = mc6[k] ^ RK6[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb7
      assign sb7[j] = sbox[st6[j]];
    end
  endgenerate

  assign sr7[0] = sb7[0];
  assign sr7[1] = sb7[5];
  assign sr7[2] = sb7[10];
  assign sr7[3] = sb7[15];
  assign sr7[4] = sb7[4];
  assign sr7[5] = sb7[9];
  assign sr7[6] = sb7[14];
  assign sr7[7] = sb7[3];
  assign sr7[8] = sb7[8];
  assign sr7[9] = sb7[13];
  assign sr7[10] = sb7[2];
  assign sr7[11] = sb7[7];
  assign sr7[12] = sb7[12];
  assign sr7[13] = sb7[1];
  assign sr7[14] = sb7[6];
  assign sr7[15] = sb7[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc7
      wire [31:0] col = mixcol({sr7[c * 4 + 0], sr7[c * 4 + 1], sr7[c * 4 + 2], sr7[c * 4 + 3]});
      assign mc7[c * 4 + 0] = col[31:24];
      assign mc7[c * 4 + 1] = col[23:16];
      assign mc7[c * 4 + 2] = col[15:8];
      assign mc7[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add7
      assign st7[k] = mc7[k] ^ RK7[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb8
      assign sb8[j] = sbox[st7[j]];
    end
  endgenerate

  assign sr8[0] = sb8[0];
  assign sr8[1] = sb8[5];
  assign sr8[2] = sb8[10];
  assign sr8[3] = sb8[15];
  assign sr8[4] = sb8[4];
  assign sr8[5] = sb8[9];
  assign sr8[6] = sb8[14];
  assign sr8[7] = sb8[3];
  assign sr8[8] = sb8[8];
  assign sr8[9] = sb8[13];
  assign sr8[10] = sb8[2];
  assign sr8[11] = sb8[7];
  assign sr8[12] = sb8[12];
  assign sr8[13] = sb8[1];
  assign sr8[14] = sb8[6];
  assign sr8[15] = sb8[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc8
      wire [31:0] col = mixcol({sr8[c * 4 + 0], sr8[c * 4 + 1], sr8[c * 4 + 2], sr8[c * 4 + 3]});
      assign mc8[c * 4 + 0] = col[31:24];
      assign mc8[c * 4 + 1] = col[23:16];
      assign mc8[c * 4 + 2] = col[15:8];
      assign mc8[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add8
      assign st8[k] = mc8[k] ^ RK8[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_sb9
      assign sb9[j] = sbox[st8[j]];
    end
  endgenerate

  assign sr9[0] = sb9[0];
  assign sr9[1] = sb9[5];
  assign sr9[2] = sb9[10];
  assign sr9[3] = sb9[15];
  assign sr9[4] = sb9[4];
  assign sr9[5] = sb9[9];
  assign sr9[6] = sb9[14];
  assign sr9[7] = sb9[3];
  assign sr9[8] = sb9[8];
  assign sr9[9] = sb9[13];
  assign sr9[10] = sb9[2];
  assign sr9[11] = sb9[7];
  assign sr9[12] = sb9[12];
  assign sr9[13] = sb9[1];
  assign sr9[14] = sb9[6];
  assign sr9[15] = sb9[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc9
      wire [31:0] col = mixcol({sr9[c * 4 + 0], sr9[c * 4 + 1], sr9[c * 4 + 2], sr9[c * 4 + 3]});
      assign mc9[c * 4 + 0] = col[31:24];
      assign mc9[c * 4 + 1] = col[23:16];
      assign mc9[c * 4 + 2] = col[15:8];
      assign mc9[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add9
      assign st9[k] = mc9[k] ^ RK9[127 - (k * 8) -: 8];
    end
  endgenerate

  genvar j10;
  generate
    for (j10 = 0; j10 < 16; j10 = j10 + 1) begin : gen_sb10
      assign sb10[j10] = sbox[st9[j10]];
    end
  endgenerate

  assign sr10[0] = sb10[0];
  assign sr10[1] = sb10[5];
  assign sr10[2] = sb10[10];
  assign sr10[3] = sb10[15];
  assign sr10[4] = sb10[4];
  assign sr10[5] = sb10[9];
  assign sr10[6] = sb10[14];
  assign sr10[7] = sb10[3];
  assign sr10[8] = sb10[8];
  assign sr10[9] = sb10[13];
  assign sr10[10] = sb10[2];
  assign sr10[11] = sb10[7];
  assign sr10[12] = sb10[12];
  assign sr10[13] = sb10[1];
  assign sr10[14] = sb10[6];
  assign sr10[15] = sb10[11];

  genvar k10;
  generate
    for (k10 = 0; k10 < 16; k10 = k10 + 1) begin : gen_add10
      assign st10[k10] = sr10[k10] ^ RK10[127 - (k10 * 8) -: 8];
    end
  endgenerate

  assign ct = {
    st10[0],
    st10[1],
    st10[2],
    st10[3],
    st10[4],
    st10[5],
    st10[6],
    st10[7],
    st10[8],
    st10[9],
    st10[10],
    st10[11],
    st10[12],
    st10[13],
    st10[14],
    st10[15]
  };
endmodule
