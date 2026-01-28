module lfsr_table_W64 (
  input  wire        clk,
  input  wire        rst,
  input  wire [63:0] seed,
  output wire [63:0] acc_out
);
  reg [63:0] lfsr;
  reg [63:0] acc;
  wire [63:0] lfsr_next;
  wire [63:0] mixed;

  assign lfsr_next = {lfsr[62:0], lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59]};

  assign mixed =
    (lfsr[4:0] == 5'd0)  ? (lfsr ^ 64'h0123456789abcdef) :
    (lfsr[4:0] == 5'd1)  ? (lfsr ^ 64'hfedcba9876543210) :
    (lfsr[4:0] == 5'd2)  ? (lfsr ^ 64'h1111111111111111) :
    (lfsr[4:0] == 5'd3)  ? (lfsr ^ 64'h2222222222222222) :
    (lfsr[4:0] == 5'd4)  ? (lfsr ^ 64'h3333333333333333) :
    (lfsr[4:0] == 5'd5)  ? (lfsr ^ 64'h4444444444444444) :
    (lfsr[4:0] == 5'd6)  ? (lfsr ^ 64'h5555555555555555) :
    (lfsr[4:0] == 5'd7)  ? (lfsr ^ 64'h6666666666666666) :
    (lfsr[4:0] == 5'd8)  ? (lfsr ^ 64'h7777777777777777) :
    (lfsr[4:0] == 5'd9)  ? (lfsr ^ 64'h8888888888888888) :
    (lfsr[4:0] == 5'd10) ? (lfsr ^ 64'h9999999999999999) :
    (lfsr[4:0] == 5'd11) ? (lfsr ^ 64'haaaaaaaaaaaaaaaa) :
    (lfsr[4:0] == 5'd12) ? (lfsr ^ 64'hbbbbbbbbbbbbbbbb) :
    (lfsr[4:0] == 5'd13) ? (lfsr ^ 64'hcccccccccccccccc) :
    (lfsr[4:0] == 5'd14) ? (lfsr ^ 64'hdddddddddddddddd) :
    (lfsr[4:0] == 5'd15) ? (lfsr ^ 64'heeeeeeeeeeeeeeee) :
    (lfsr[4:0] == 5'd16) ? (lfsr ^ 64'hffffffffffffffff) :
    (lfsr[4:0] == 5'd17) ? (lfsr ^ 64'h0f0f0f0f0f0f0f0f) :
    (lfsr[4:0] == 5'd18) ? (lfsr ^ 64'hf0f0f0f0f0f0f0f0) :
    (lfsr[4:0] == 5'd19) ? (lfsr ^ 64'h00ff00ff00ff00ff) :
    (lfsr[4:0] == 5'd20) ? (lfsr ^ 64'hff00ff00ff00ff00) :
    (lfsr[4:0] == 5'd21) ? (lfsr ^ 64'h0000ffff0000ffff) :
    (lfsr[4:0] == 5'd22) ? (lfsr ^ 64'hffff0000ffff0000) :
    (lfsr[4:0] == 5'd23) ? (lfsr ^ 64'h00000000ffffffff) :
    (lfsr[4:0] == 5'd24) ? (lfsr ^ 64'hffffffff00000000) :
    (lfsr[4:0] == 5'd25) ? (lfsr ^ 64'h123456789abcdef0) :
    (lfsr[4:0] == 5'd26) ? (lfsr ^ 64'h0fedcba987654321) :
    (lfsr[4:0] == 5'd27) ? (lfsr ^ 64'ha5a5a5a5a5a5a5a5) :
    (lfsr[4:0] == 5'd28) ? (lfsr ^ 64'h5a5a5a5a5a5a5a5a) :
    (lfsr[4:0] == 5'd29) ? (lfsr ^ 64'hc3c3c3c3c3c3c3c3) :
    (lfsr[4:0] == 5'd30) ? (lfsr ^ 64'h3c3c3c3c3c3c3c3c) :
    (lfsr ^ 64'h8181818181818181);

  assign acc_out = acc;

  always @(posedge clk) begin
    if (rst) begin
      lfsr <= seed;
      acc <= 0;
    end
    else begin
      lfsr <= lfsr_next;
      acc <= acc ^ mixed;
    end
  end
endmodule
