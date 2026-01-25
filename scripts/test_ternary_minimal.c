// Minimal test of _mm512_ternarylogic_epi64 instruction
// Tests that the bit ordering matches our expectations

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

int main() {
    printf("Testing _mm512_ternarylogic_epi64 bit ordering\n\n");

    // Test XOR3 (a ^ b ^ c) - imm8 = 0x96
    // Truth table:
    //   a b c | result
    //   0 0 0 |   0
    //   0 0 1 |   1
    //   0 1 0 |   1
    //   0 1 1 |   0
    //   1 0 0 |   1
    //   1 0 1 |   0
    //   1 1 0 |   0
    //   1 1 1 |   1
    // So imm8 = 0b10010110 = 0x96

    printf("Test 1: XOR3 function (imm8 = 0x96)\n");
    printf("  Intrinsic uses: index = (c << 2) | (b << 1) | a\n");
    printf("  Circuit uses:   index = (a << 2) | (b << 1) | c\n\n");

    // Set up test vectors
    // a=1, b=0, c=0 should give result=1 (for XOR3)
    __m512i a = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFULL);  // all 1s
    __m512i b = _mm512_set1_epi64(0x0000000000000000ULL);  // all 0s
    __m512i c = _mm512_set1_epi64(0x0000000000000000ULL);  // all 0s

    // Using intrinsic: _mm512_ternarylogic_epi64(a, b, c, imm8)
    // Intrinsic computes: index = (c << 2) | (b << 1) | a
    // For a=1, b=0, c=0: index = (0 << 2) | (0 << 1) | 1 = 1
    // imm8[1] = bit 1 of 0x96 = 1
    // Expected: result = 1
    __m512i result1 = _mm512_ternarylogic_epi64(a, b, c, 0x96);
    uint64_t val1 = _mm_cvtsi128_si64(_mm512_castsi512_si128(result1));
    printf("  _mm512_ternarylogic_epi64(a=1, b=0, c=0, 0x96):\n");
    printf("    Intrinsic index = (c << 2) | (b << 1) | a = (0 << 2) | (0 << 1) | 1 = 1\n");
    printf("    imm8[1] = bit 1 of 0x96 = %d\n", (0x96 >> 1) & 1);
    printf("    Result = %s (all bits same: %d)\n", val1 ? "1" : "0", (val1 & 1) == ((val1 >> 63) & 1));
    printf("    Expected for XOR(1, 0, 0) = 1\n\n");

    // Now test with swapped operand order: (c, b, a)
    // This should give index = (a << 2) | (b << 1) | c = (1 << 2) | (0 << 1) | 0 = 4
    // imm8[4] = bit 4 of 0x96 = 1
    __m512i result2 = _mm512_ternarylogic_epi64(c, b, a, 0x96);
    uint64_t val2 = _mm_cvtsi128_si64(_mm512_castsi512_si128(result2));
    printf("  _mm512_ternarylogic_epi64(c=0, b=0, a=1, 0x96) [swapped a and c]:\n");
    printf("    Intrinsic index = (a << 2) | (b << 1) | c = (1 << 2) | (0 << 1) | 0 = 4\n");
    printf("    imm8[4] = bit 4 of 0x96 = %d\n", (0x96 >> 4) & 1);
    printf("    Result = %s\n", val2 ? "1" : "0");
    printf("    Expected for XOR(1, 0, 0) = 1\n\n");

    // Test another case: a=1, b=1, c=0
    // XOR(1, 1, 0) = 0
    a = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFULL);
    b = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFULL);
    c = _mm512_set1_epi64(0x0000000000000000ULL);

    // Original order: index = (0 << 2) | (1 << 1) | 1 = 3, imm8[3] = 0
    result1 = _mm512_ternarylogic_epi64(a, b, c, 0x96);
    val1 = _mm_cvtsi128_si64(_mm512_castsi512_si128(result1));
    printf("  _mm512_ternarylogic_epi64(a=1, b=1, c=0, 0x96):\n");
    printf("    Intrinsic index = (0 << 2) | (1 << 1) | 1 = 3, imm8[3] = %d\n", (0x96 >> 3) & 1);
    printf("    Result = %s, Expected XOR(1,1,0) = 0\n\n", val1 ? "1" : "0");

    // Swapped order: index = (1 << 2) | (1 << 1) | 0 = 6, imm8[6] = 0
    result2 = _mm512_ternarylogic_epi64(c, b, a, 0x96);
    val2 = _mm_cvtsi128_si64(_mm512_castsi512_si128(result2));
    printf("  _mm512_ternarylogic_epi64(c=0, b=1, a=1, 0x96) [swapped]:\n");
    printf("    Intrinsic index = (1 << 2) | (1 << 1) | 0 = 6, imm8[6] = %d\n", (0x96 >> 6) & 1);
    printf("    Result = %s, Expected XOR(1,1,0) = 0\n\n", val2 ? "1" : "0");

    printf("Summary:\n");
    printf("  To get result = F(a, b, c) where circuit uses idx = (a << 2) | (b << 1) | c:\n");
    printf("  Call _mm512_ternarylogic_epi64(c, b, a, imm8) to swap a and c\n");
    printf("  Then intrinsic computes idx = (a << 2) | (b << 1) | c which matches circuit\n");

    return 0;
}
