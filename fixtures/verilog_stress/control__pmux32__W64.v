module pmux32 (
  input  wire [4:0]  sel,
  input  wire [63:0] a0,
  input  wire [63:0] a1,
  input  wire [63:0] a2,
  input  wire [63:0] a3,
  input  wire [63:0] a4,
  input  wire [63:0] a5,
  input  wire [63:0] a6,
  input  wire [63:0] a7,
  input  wire [63:0] a8,
  input  wire [63:0] a9,
  input  wire [63:0] a10,
  input  wire [63:0] a11,
  input  wire [63:0] a12,
  input  wire [63:0] a13,
  input  wire [63:0] a14,
  input  wire [63:0] a15,
  input  wire [63:0] a16,
  input  wire [63:0] a17,
  input  wire [63:0] a18,
  input  wire [63:0] a19,
  input  wire [63:0] a20,
  input  wire [63:0] a21,
  input  wire [63:0] a22,
  input  wire [63:0] a23,
  input  wire [63:0] a24,
  input  wire [63:0] a25,
  input  wire [63:0] a26,
  input  wire [63:0] a27,
  input  wire [63:0] a28,
  input  wire [63:0] a29,
  input  wire [63:0] a30,
  input  wire [63:0] a31,
  output reg  [63:0] y
);
  always @* begin
    case (sel)
      5'd0: y=a0;
      5'd1: y=a1;
      5'd2: y=a2;
      5'd3: y=a3;
      5'd4: y=a4;
      5'd5: y=a5;
      5'd6: y=a6;
      5'd7: y=a7;
      5'd8: y=a8;
      5'd9: y=a9;
      5'd10:y=a10;
      5'd11:y=a11;
      5'd12:y=a12;
      5'd13:y=a13;
      5'd14:y=a14;
      5'd15:y=a15;
      5'd16:y=a16;
      5'd17:y=a17;
      5'd18:y=a18;
      5'd19:y=a19;
      5'd20:y=a20;
      5'd21:y=a21;
      5'd22:y=a22;
      5'd23:y=a23;
      5'd24:y=a24;
      5'd25:y=a25;
      5'd26:y=a26;
      5'd27:y=a27;
      5'd28:y=a28;
      5'd29:y=a29;
      5'd30:y=a30;
      default: y=a31;
    endcase
  end
endmodule
