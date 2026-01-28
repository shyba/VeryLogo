module vec_add #(parameter N=1048576) (
  input  wire [32*N-1:0] a,
  input  wire [32*N-1:0] b,
  output wire [32*N-1:0] c
);
  genvar i;
  generate
    for (i=0; i<N; i=i+1) begin : G
      assign c[32*i +: 32] = a[32*i +: 32] + b[32*i +: 32];
    end
  endgenerate
endmodule
