module aes128_fixedkey_seq_lut(
  input wire clk,
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

  reg [127:0] st;
  reg [3:0] round;

  wire [127:0] st_init = pt ^ RK0;

  wire [127:0] rk_mid = (round == 4'd1) ? RK1 :
                        (round == 4'd2) ? RK2 :
                        (round == 4'd3) ? RK3 :
                        (round == 4'd4) ? RK4 :
                        (round == 4'd5) ? RK5 :
                        (round == 4'd6) ? RK6 :
                        (round == 4'd7) ? RK7 :
                        (round == 4'd8) ? RK8 :
                        (round == 4'd9) ? RK9 :
                        RK1;

  wire [7:0] st_b [0:15];
  wire [7:0] sb [0:15];
  wire [7:0] sr [0:15];
  wire [7:0] mc [0:15];
  wire [7:0] out_mid_b [0:15];
  wire [127:0] st_mid;

  genvar i;
  generate
    for (i = 0; i < 16; i = i + 1) begin : gen_bytes
      assign st_b[i] = st[127 - (i * 8) -: 8];
      assign sb[i] = sbox[st_b[i]];
    end
  endgenerate

  assign sr[0] = sb[0];
  assign sr[1] = sb[5];
  assign sr[2] = sb[10];
  assign sr[3] = sb[15];
  assign sr[4] = sb[4];
  assign sr[5] = sb[9];
  assign sr[6] = sb[14];
  assign sr[7] = sb[3];
  assign sr[8] = sb[8];
  assign sr[9] = sb[13];
  assign sr[10] = sb[2];
  assign sr[11] = sb[7];
  assign sr[12] = sb[12];
  assign sr[13] = sb[1];
  assign sr[14] = sb[6];
  assign sr[15] = sb[11];

  genvar c;
  generate
    for (c = 0; c < 4; c = c + 1) begin : gen_mc
      wire [31:0] col = mixcol({sr[c * 4 + 0], sr[c * 4 + 1], sr[c * 4 + 2], sr[c * 4 + 3]});
      assign mc[c * 4 + 0] = col[31:24];
      assign mc[c * 4 + 1] = col[23:16];
      assign mc[c * 4 + 2] = col[15:8];
      assign mc[c * 4 + 3] = col[7:0];
    end
  endgenerate

  genvar j;
  generate
    for (j = 0; j < 16; j = j + 1) begin : gen_add_mid
      assign out_mid_b[j] = mc[j] ^ rk_mid[127 - (j * 8) -: 8];
    end
  endgenerate

  assign st_mid = {
    out_mid_b[0],
    out_mid_b[1],
    out_mid_b[2],
    out_mid_b[3],
    out_mid_b[4],
    out_mid_b[5],
    out_mid_b[6],
    out_mid_b[7],
    out_mid_b[8],
    out_mid_b[9],
    out_mid_b[10],
    out_mid_b[11],
    out_mid_b[12],
    out_mid_b[13],
    out_mid_b[14],
    out_mid_b[15]
  };

  wire [7:0] out_fin_b [0:15];
  genvar k;
  generate
    for (k = 0; k < 16; k = k + 1) begin : gen_add_fin
      assign out_fin_b[k] = sr[k] ^ RK10[127 - (k * 8) -: 8];
    end
  endgenerate

  wire [127:0] st_fin = {
    out_fin_b[0],
    out_fin_b[1],
    out_fin_b[2],
    out_fin_b[3],
    out_fin_b[4],
    out_fin_b[5],
    out_fin_b[6],
    out_fin_b[7],
    out_fin_b[8],
    out_fin_b[9],
    out_fin_b[10],
    out_fin_b[11],
    out_fin_b[12],
    out_fin_b[13],
    out_fin_b[14],
    out_fin_b[15]
  };

  always @(posedge clk) begin
    if (round == 4'd0) begin
      st <= st_init;
      round <= 4'd1;
    end else if (round < 4'd10) begin
      st <= st_mid;
      round <= round + 4'd1;
    end else if (round == 4'd10) begin
      st <= st_fin;
      round <= 4'd11;
    end else begin
      st <= st;
      round <= round;
    end
  end

  assign ct = st;
endmodule
