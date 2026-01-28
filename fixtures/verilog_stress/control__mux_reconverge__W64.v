module mux_reconverge (
  input  wire [63:0] a,
  input  wire [63:0] b,
  input  wire [63:0] c,
  input  wire [63:0] d,
  input  wire [1:0]  sel0,
  input  wire [1:0]  sel1,
  output wire [63:0] out
);
  wire [63:0] m0 = (sel0 == 2'd0) ? a :
                   (sel0 == 2'd1) ? b :
                   (sel0 == 2'd2) ? c : d;

  wire [63:0] m1 = (sel1 == 2'd0) ? a :
                   (sel1 == 2'd1) ? b :
                   (sel1 == 2'd2) ? c : d;

  wire [63:0] m2 = (sel0 == sel1) ? a : b;

  wire [63:0] m3 = (sel0[0] == sel1[0]) ? m0 : m1;

  wire [63:0] combined = m0 ^ m1;
  wire [63:0] combined2 = m2 + m3;

  assign out = combined ^ combined2;
endmodule
