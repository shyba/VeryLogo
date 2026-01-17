module top(
  input logic clk,
  input logic rst,
  input logic [127:0] pt,
  output logic [127:0] ct
);
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

  function automatic [7:0] xtime(input [7:0] x);
    reg [7:0] x1;
    begin
      x1 = ((x << 1) & 8'hff);
      xtime = x1 ^ (x[7] ? 8'h1b : 8'h00);
    end
  endfunction

  function automatic [7:0] gf_mul(input [7:0] a, input [7:0] b);
    reg [7:0] p;
    reg [7:0] aa;
    reg [7:0] bb;
    integer i;
    begin
      p = 8'h00;
      aa = a;
      bb = b;
      for (i = 0; i < 8; i = i + 1) begin
        p = p ^ (bb[0] ? aa : 8'h00);
        bb = bb >> 1;
        aa = xtime(aa);
      end
      gf_mul = p;
    end
  endfunction

  function automatic [7:0] gf_inv(input [7:0] x);
    reg [7:0] x2;
    reg [7:0] x4;
    reg [7:0] x8;
    reg [7:0] x16;
    reg [7:0] x32;
    reg [7:0] x64;
    reg [7:0] x128;
    reg [7:0] t;
    begin
      x2 = gf_mul(x, x);
      x4 = gf_mul(x2, x2);
      x8 = gf_mul(x4, x4);
      x16 = gf_mul(x8, x8);
      x32 = gf_mul(x16, x16);
      x64 = gf_mul(x32, x32);
      x128 = gf_mul(x64, x64);
      t = gf_mul(x128, x64);
      t = gf_mul(t, x32);
      t = gf_mul(t, x16);
      t = gf_mul(t, x8);
      t = gf_mul(t, x4);
      t = gf_mul(t, x2);
      gf_inv = (x == 8'h00) ? 8'h00 : t;
    end
  endfunction

  function automatic [7:0] rotl1(input [7:0] x);
    begin
      rotl1 = (((x << 1) | (x >> 7)) & 8'hff);
    end
  endfunction

  function automatic [7:0] rotl2(input [7:0] x);
    begin
      rotl2 = (((x << 2) | (x >> 6)) & 8'hff);
    end
  endfunction

  function automatic [7:0] rotl3(input [7:0] x);
    begin
      rotl3 = (((x << 3) | (x >> 5)) & 8'hff);
    end
  endfunction

  function automatic [7:0] rotl4(input [7:0] x);
    begin
      rotl4 = (((x << 4) | (x >> 4)) & 8'hff);
    end
  endfunction

  function automatic [7:0] affine(input [7:0] x);
    begin
      affine = x ^ rotl1(x) ^ rotl2(x) ^ rotl3(x) ^ rotl4(x) ^ 8'h63;
    end
  endfunction

  function automatic [7:0] sbox(input [7:0] x);
    begin
      sbox = affine(gf_inv(x));
    end
  endfunction

  function automatic [31:0] mixcol(input [31:0] c);
    reg [7:0] a0;
    reg [7:0] a1;
    reg [7:0] a2;
    reg [7:0] a3;
    reg [7:0] t;
    reg [7:0] u;
    reg [7:0] b0;
    reg [7:0] b1;
    reg [7:0] b2;
    reg [7:0] b3;
    begin
      a0 = c[31:24];
      a1 = c[23:16];
      a2 = c[15:8];
      a3 = c[7:0];
      t = a0 ^ a1 ^ a2 ^ a3;
      u = a0;
      b0 = a0 ^ t ^ xtime(a0 ^ a1);
      b1 = a1 ^ t ^ xtime(a1 ^ a2);
      b2 = a2 ^ t ^ xtime(a2 ^ a3);
      b3 = a3 ^ t ^ xtime(a3 ^ u);
      mixcol = {b0, b1, b2, b3};
    end
  endfunction

  function automatic [127:0] sub_bytes(input [127:0] s);
    reg [127:0] o;
    begin
      o[127:120] = sbox(s[127:120]);
      o[119:112] = sbox(s[119:112]);
      o[111:104] = sbox(s[111:104]);
      o[103:96] = sbox(s[103:96]);
      o[95:88] = sbox(s[95:88]);
      o[87:80] = sbox(s[87:80]);
      o[79:72] = sbox(s[79:72]);
      o[71:64] = sbox(s[71:64]);
      o[63:56] = sbox(s[63:56]);
      o[55:48] = sbox(s[55:48]);
      o[47:40] = sbox(s[47:40]);
      o[39:32] = sbox(s[39:32]);
      o[31:24] = sbox(s[31:24]);
      o[23:16] = sbox(s[23:16]);
      o[15:8] = sbox(s[15:8]);
      o[7:0] = sbox(s[7:0]);
      sub_bytes = o;
    end
  endfunction

  function automatic [127:0] shift_rows(input [127:0] s);
    reg [7:0] b0;
    reg [7:0] b1;
    reg [7:0] b2;
    reg [7:0] b3;
    reg [7:0] b4;
    reg [7:0] b5;
    reg [7:0] b6;
    reg [7:0] b7;
    reg [7:0] b8;
    reg [7:0] b9;
    reg [7:0] b10;
    reg [7:0] b11;
    reg [7:0] b12;
    reg [7:0] b13;
    reg [7:0] b14;
    reg [7:0] b15;
    reg [127:0] o;
    begin
      b0 = s[127:120];
      b1 = s[119:112];
      b2 = s[111:104];
      b3 = s[103:96];
      b4 = s[95:88];
      b5 = s[87:80];
      b6 = s[79:72];
      b7 = s[71:64];
      b8 = s[63:56];
      b9 = s[55:48];
      b10 = s[47:40];
      b11 = s[39:32];
      b12 = s[31:24];
      b13 = s[23:16];
      b14 = s[15:8];
      b15 = s[7:0];

      o[127:120] = b0;
      o[119:112] = b5;
      o[111:104] = b10;
      o[103:96] = b15;
      o[95:88] = b4;
      o[87:80] = b9;
      o[79:72] = b14;
      o[71:64] = b3;
      o[63:56] = b8;
      o[55:48] = b13;
      o[47:40] = b2;
      o[39:32] = b7;
      o[31:24] = b12;
      o[23:16] = b1;
      o[15:8] = b6;
      o[7:0] = b11;

      shift_rows = o;
    end
  endfunction

  function automatic [127:0] mix_columns(input [127:0] s);
    reg [31:0] c0;
    reg [31:0] c1;
    reg [31:0] c2;
    reg [31:0] c3;
    reg [127:0] o;
    begin
      c0 = {s[127:120], s[119:112], s[111:104], s[103:96]};
      c1 = {s[95:88], s[87:80], s[79:72], s[71:64]};
      c2 = {s[63:56], s[55:48], s[47:40], s[39:32]};
      c3 = {s[31:24], s[23:16], s[15:8], s[7:0]};
      c0 = mixcol(c0);
      c1 = mixcol(c1);
      c2 = mixcol(c2);
      c3 = mixcol(c3);
      o[127:96] = c0;
      o[95:64] = c1;
      o[63:32] = c2;
      o[31:0] = c3;
      mix_columns = o;
    end
  endfunction

  logic [3:0] round;
  logic [127:0] st;

  wire [127:0] rk_full =
    (round == 4'd1) ? RK1 :
    (round == 4'd2) ? RK2 :
    (round == 4'd3) ? RK3 :
    (round == 4'd4) ? RK4 :
    (round == 4'd5) ? RK5 :
    (round == 4'd6) ? RK6 :
    (round == 4'd7) ? RK7 :
    (round == 4'd8) ? RK8 :
    RK9;

  wire [127:0] round_full = mix_columns(shift_rows(sub_bytes(st))) ^ rk_full;
  wire [127:0] round_final = shift_rows(sub_bytes(st)) ^ RK10;

  wire is_r0 = (round == 4'd0);
  wire is_r10 = (round == 4'd10);
  wire is_r_done = (round >= 4'd11);

  wire [127:0] st_next =
    is_r0 ? (pt ^ RK0) :
    is_r10 ? round_final :
    is_r_done ? st :
    round_full;

  wire [3:0] round_next =
    is_r0 ? 4'd1 :
    is_r10 ? 4'd11 :
    is_r_done ? round :
    (round + 4'd1);

  always_ff @(posedge clk) begin
    if (rst) begin
      round <= 4'd0;
      st <= 128'd0;
    end else begin
      round <= round_next;
      st <= st_next;
    end
  end

  assign ct = st;
endmodule

