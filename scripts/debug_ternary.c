// Debug version - trace first few gates for input=1
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

int main() {
    printf("Debugging ternary circuit for input=1\n\n");

    // Set up for input=1 (only bit 0 is set)
    __m512i in[8];
    in[0] = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFULL);  // bit 0 = 1
    for (int i = 1; i < 8; i++) {
        in[i] = _mm512_set1_epi64(0x0000000000000000ULL);
    }

    __m512i ones = _mm512_set1_epi64(-1);
    __m512i t[100];  // Only need first 100 for debugging

    // Gate 0: const(1, 1)
    t[0] = ones;
    printf("t[0] (const 1): %016lx\n", _mm_cvtsi128_si64(_mm512_castsi512_si128(t[0])) & 1);

    // Gate 1: xor(0, 8) -> xor(in[0], t[0])
    t[1] = _mm512_xor_si512(in[0], t[0]);
    printf("t[1] (in[0] xor t[0]): %016lx\n", _mm_cvtsi128_si64(_mm512_castsi512_si128(t[1])) & 1);
    printf("  Expected: in[0]=1, t[0]=1, result = 1^1 = 0\n");

    // Gate 2: and(0, 1) -> and(in[0], in[1])
    t[2] = _mm512_and_si512(in[0], in[1]);
    printf("t[2] (in[0] and in[1]): %016lx\n", _mm_cvtsi128_si64(_mm512_castsi512_si128(t[2])) & 1);
    printf("  Expected: in[0]=1, in[1]=0, result = 1&0 = 0\n");

    // Gate 3: xor(2, 10) -> xor(in[2], t[2])
    t[3] = _mm512_xor_si512(in[2], t[2]);
    printf("t[3] (in[2] xor t[2]): %016lx\n", _mm_cvtsi128_si64(_mm512_castsi512_si128(t[3])) & 1);
    printf("  Expected: in[2]=0, t[2]=0, result = 0^0 = 0\n");

    // Gate 4: and(1, 2) -> and(in[1], in[2])
    t[4] = _mm512_and_si512(in[1], in[2]);
    printf("t[4] (in[1] and in[2]): %016lx\n", _mm_cvtsi128_si64(_mm512_castsi512_si128(t[4])) & 1);

    // Let me also trace what Python says
    printf("\n\nPython evaluation for input=1:\n");
    printf("Should produce S-box[1] = 0x7c\n");
    printf("But C code produces something else...\n");

    // Now let me compare in[0]..in[7] to what Python expects
    printf("\nInput bit planes for input=1:\n");
    for (int bit = 0; bit < 8; bit++) {
        uint64_t val = _mm_cvtsi128_si64(_mm512_castsi512_si128(in[bit]));
        printf("  in[%d] = %lx (bit %d of input 1 = %d)\n", bit, val, bit, (1 >> bit) & 1);
    }

    return 0;
}
