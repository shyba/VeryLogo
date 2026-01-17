module top(
  input logic clk,
  input logic en,
  input logic [3:0] d,
  output logic [3:0] q
);
  always_ff @(posedge clk) begin
    if (en) q <= d;
  end
endmodule

