module xor_diamonds #(parameter D=6) (
  input  wire [63:0] seed,
  output wire [63:0] result
);

  function [63:0] diamond_node;
    input [63:0] a;
    input [63:0] b;
    input [63:0] c;
    input [63:0] d;
    begin
      diamond_node = ((a ^ b) ^ (c ^ d)) ^ ((a ^ c) ^ (b ^ d));
    end
  endfunction

  wire [63:0] level[0:D];

  assign level[0] = seed;

  genvar i;
  generate
    for (i=0; i<D; i=i+1) begin : diamond_gen
      wire [63:0] x = level[i];
      wire [63:0] rot1 = {x[62:0], x[63]};
      wire [63:0] rot2 = {x[61:0], x[63:62]};
      wire [63:0] rot3 = {x[60:0], x[63:61]};
      wire [63:0] rot4 = {x[59:0], x[63:60]};

      wire [63:0] d1 = diamond_node(x, rot1, rot2, rot3);
      wire [63:0] d2 = diamond_node(rot1, rot2, rot3, rot4);
      wire [63:0] d3 = diamond_node(x, rot2, rot4, d1);
      wire [63:0] d4 = diamond_node(rot1, rot3, d1, d2);

      wire [63:0] reconverge1 = d1 ^ d2 ^ d3;
      wire [63:0] reconverge2 = d2 ^ d3 ^ d4;
      wire [63:0] reconverge3 = d1 ^ d3 ^ d4;
      wire [63:0] reconverge4 = d1 ^ d2 ^ d4;

      assign level[i+1] = reconverge1 ^ reconverge2 ^ reconverge3 ^ reconverge4;
    end
  endgenerate

  assign result = level[D];

endmodule

