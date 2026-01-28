module roundN (
  input  wire        clk,
  input  wire        rst,
  input  wire [63:0] in,
  output wire [63:0] out
);
  reg [63:0] s;
  reg [31:0] r;
  function [63:0] f(input [63:0] x, input [7:0] k);
    begin
      f = {x[62:0],x[63]} ^ (x + {{56{1'b0}}, k});
    end
  endfunction
  always @(posedge clk) begin
    if (rst) begin s <= in; r <= 0; end
    else if (r < 8) begin s <= f(s,r[7:0]); r <= r+1; end
  end

  assign out = s;
endmodule
