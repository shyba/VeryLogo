#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "aes_sbox_table.h"
#include "../out/test_math_synth/circuit_avx512.c"

int main() {
    printf("AVX-512 Mathematical Tower Field S-box Verification\n");
    printf("===================================================\n\n");

    int errors = 0;

    for (int batch = 0; batch < 4; batch++) {
        __m512i in_planes[8];
        __m512i out_planes[8];

        for (int bit = 0; bit < 8; bit++) {
            uint64_t lanes[8] = {0};
            for (int i = 0; i < 64; i++) {
                int val = batch * 64 + i;
                if ((val >> bit) & 1) {
                    int lane_idx = i / 8;
                    int bit_in_lane = i % 8;
                    lanes[lane_idx] |= (1ULL << (bit_in_lane * 8));
                }
            }
            in_planes[bit] = _mm512_set_epi64(
                lanes[7], lanes[6], lanes[5], lanes[4],
                lanes[3], lanes[2], lanes[1], lanes[0]
            );
        }

        circuit__core(in_planes, nullptr, out_planes, nullptr);

        alignas(64) uint64_t out_lanes[8];
        for (int i = 0; i < 64; i++) {
            uint8_t computed = 0;
            for (int bit = 0; bit < 8; bit++) {
                _mm512_store_si512((__m512i*)out_lanes, out_planes[bit]);
                int lane_idx = i / 8;
                int bit_in_lane = i % 8;
                if (out_lanes[lane_idx] & (1ULL << (bit_in_lane * 8))) {
                    computed |= (1 << bit);
                }
            }

            int input_val = batch * 64 + i;
            uint8_t expected = AES_SBOX[input_val];

            if (computed != expected) {
                printf("ERROR: S-box(0x%02x) = 0x%02x, expected 0x%02x\n",
                       input_val, computed, expected);
                errors++;
            }
        }

        printf("Batch %d (0x%02x-0x%02x): ", batch, batch * 64, batch * 64 + 63);
        if (errors == 0) {
            printf("✓ 64/64 correct\n");
        } else {
            printf("✗ Errors detected\n");
        }
    }

    printf("\n");
    if (errors == 0) {
        printf("✅ All 256 S-box values correct!\n\n");
        printf("Mathematical tower field S-box compiled via Yosys synth + ABC is computing correctly.\n");
        return 0;
    } else {
        printf("❌ %d errors found!\n", errors);
        return 1;
    }
}
