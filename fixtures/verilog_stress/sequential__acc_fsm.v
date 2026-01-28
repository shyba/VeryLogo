module acc_fsm (
  input  wire        clk,
  input  wire        rst,
  input  wire [31:0] limit,
  input  wire [31:0] x,
  output reg  [31:0] acc
);
  reg [31:0] i;
  always @(posedge clk) begin
    if (rst) begin acc <= 0; i <= 0; end
    else if (i < limit) begin
      acc <= acc + (x ^ i);
      i <= i + 1;
    end
  end
endmodule
