module tower_sbox(
    input wire [7:0] in_byte,
    output wire [7:0] out_byte
);

    // Top linear layer (27 XOR gates)
    wire T1 = in_byte[7] ^ in_byte[4];
    wire T2 = in_byte[7] ^ in_byte[2];
    wire T3 = in_byte[7] ^ in_byte[1];
    wire T4 = in_byte[4] ^ in_byte[2];
    wire T5 = in_byte[3] ^ in_byte[1];
    wire T6 = T1 ^ T5;
    wire T7 = in_byte[6] ^ in_byte[5];
    wire T8 = in_byte[0] ^ T6;
    wire T9 = in_byte[0] ^ T7;
    wire T10 = T6 ^ T7;
    wire T11 = in_byte[6] ^ in_byte[2];
    wire T12 = in_byte[5] ^ in_byte[2];
    wire T13 = T3 ^ T4;
    wire T14 = T6 ^ T11;
    wire T15 = T5 ^ T11;
    wire T16 = T5 ^ T12;
    wire T17 = T9 ^ T16;
    wire T18 = in_byte[4] ^ in_byte[0];
    wire T19 = T7 ^ T18;
    wire T20 = T1 ^ T19;
    wire T21 = in_byte[1] ^ in_byte[0];
    wire T22 = T7 ^ T21;
    wire T23 = T2 ^ T22;
    wire T24 = T2 ^ T10;
    wire T25 = T20 ^ T17;
    wire T26 = T3 ^ T16;
    wire T27 = T1 ^ T12;

    // Non-linear middle (32 AND gates)
    wire M1 = T13 & T6;
    wire M2 = T23 & T8;
    wire M3 = T14 ^ M1;
    wire M4 = T19 & in_byte[0];
    wire M5 = M4 ^ M1;
    wire M6 = T3 & T16;
    wire M7 = T22 & T9;
    wire M8 = T26 ^ M6;
    wire M9 = T20 & T17;
    wire M10 = M9 ^ M6;
    wire M11 = T1 & T15;
    wire M12 = T4 & T27;
    wire M13 = M12 ^ M11;
    wire M14 = T2 & T10;
    wire M15 = M14 ^ M11;
    wire M16 = M3 ^ M2;
    wire M17 = M5 ^ T24;
    wire M18 = M8 ^ M7;
    wire M19 = M10 ^ M15;
    wire M20 = M16 ^ M13;
    wire M21 = M17 ^ M15;
    wire M22 = M18 ^ M13;
    wire M23 = M19 ^ T25;
    wire M24 = M22 ^ M23;
    wire M25 = M22 & M20;
    wire M26 = M21 ^ M25;
    wire M27 = M20 ^ M21;
    wire M28 = M23 ^ M25;
    wire M29 = M28 & M27;
    wire M30 = M26 & M24;
    wire M31 = M20 & M23;
    wire M32 = M27 & M31;
    wire M33 = M27 ^ M25;
    wire M34 = M21 & M22;
    wire M35 = M24 & M34;
    wire M36 = M24 ^ M25;
    wire M37 = M21 ^ M29;
    wire M38 = M32 ^ M33;
    wire M39 = M23 ^ M30;
    wire M40 = M35 ^ M36;
    wire M41 = M38 ^ M40;
    wire M42 = M37 ^ M39;
    wire M43 = M37 ^ M38;
    wire M44 = M39 ^ M40;
    wire M45 = M42 ^ M41;
    wire M46 = M44 & T6;
    wire M47 = M40 & T8;
    wire M48 = M39 & in_byte[0];
    wire M49 = M43 & T16;
    wire M50 = M38 & T9;
    wire M51 = M37 & T17;
    wire M52 = M42 & T15;
    wire M53 = M45 & T27;
    wire M54 = M41 & T10;
    wire M55 = M44 & T13;
    wire M56 = M40 & T23;
    wire M57 = M39 & T19;
    wire M58 = M43 & T3;
    wire M59 = M38 & T22;
    wire M60 = M37 & T20;
    wire M61 = M42 & T1;
    wire M62 = M45 & T4;
    wire M63 = M41 & T2;

    // Bottom linear layer (69 XOR gates)
    wire L0 = M61 ^ M62;
    wire L1 = M50 ^ M56;
    wire L2 = M46 ^ M48;
    wire L3 = M47 ^ M55;
    wire L4 = M54 ^ M58;
    wire L5 = M49 ^ M61;
    wire L6 = M62 ^ L5;
    wire L7 = M46 ^ L3;
    wire L8 = M51 ^ M59;
    wire L9 = M52 ^ M53;
    wire L10 = M53 ^ L4;
    wire L11 = M60 ^ L2;
    wire L12 = M48 ^ M51;
    wire L13 = M50 ^ L0;
    wire L14 = M52 ^ M61;
    wire L15 = M55 ^ L1;
    wire L16 = M56 ^ L0;
    wire L17 = M57 ^ L1;
    wire L18 = M58 ^ L8;
    wire L19 = M63 ^ L4;
    wire L20 = L0 ^ L1;
    wire L21 = L1 ^ L7;
    wire L22 = L3 ^ L12;
    wire L23 = L18 ^ L2;
    wire L24 = L15 ^ L9;
    wire L25 = L6 ^ L10;
    wire L26 = L7 ^ L9;
    wire L27 = L8 ^ L10;
    wire L28 = L11 ^ L14;
    wire L29 = L11 ^ L17;

    // Final outputs
    wire S7 = L6 ^ L24;
    wire S6 = L16 ^ L26;
    wire S5 = L19 ^ L28;
    wire S4 = L6 ^ L21;
    wire S3 = L20 ^ L22;
    wire S2 = L25 ^ L29;
    wire S1 = L13 ^ L27;
    wire S0 = L6 ^ L23;

    // Combinational outputs with affine constant 0x63 = 01100011
    // Inversions on bits 0, 1, 5, 6
    assign out_byte[0] = ~S0;  // inverted
    assign out_byte[1] = ~S1;  // inverted
    assign out_byte[2] = S2;
    assign out_byte[3] = S3;
    assign out_byte[4] = S4;
    assign out_byte[5] = ~S5;  // inverted
    assign out_byte[6] = ~S6;  // inverted
    assign out_byte[7] = S7;

endmodule
