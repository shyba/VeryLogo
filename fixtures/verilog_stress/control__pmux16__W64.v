module pmux16 (
  input  wire [3:0]  sel,
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
  output reg  [63:0] y
);
  always @* begin
    case (sel)
      4'd0: y=a0;
      4'd1: y=a1;
      4'd2: y=a2;
      4'd3: y=a3;
      4'd4: y=a4;
      4'd5: y=a5;
      4'd6: y=a6;
      4'd7: y=a7;
      4'd8: y=a8;
      4'd9: y=a9;
      4'd10:y=a10;
      4'd11:y=a11;
      4'd12:y=a12;
      4'd13:y=a13;
      4'd14:y=a14;
      default: y=a15;
    endcase
  end
endmodule
