/**
 * Verify AVX-512 tower field S-box against AES truth table.
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "aes_sbox_table.h"

#include "../out/test_math_synth/circuit_avx512.c"
// Forward declaration - will be linked from generated code
void circuit(__m512i* in, __m512i* out);

static void bytes_to_bitplanes(const uint8_t* bytes, __m512i* bitplanes, int count) {
    // Convert byte array to AVX-512 bit-planes
    // Each bit-plane holds 64 values (512 bits / 8 bits per value)
    for (int bit = 0; bit < 8; bit++) {
        uint64_t lanes[8] = {0};

        for (int i = 0; i < count && i < 64; i++) {
            if (bytes[i] & (1 << bit)) {
                int lane = i / 8;
                int pos = i % 8;
                lanes[lane] |= (1ULL << (pos * 8));
            }
        }

        bitplanes[bit] = _mm512_set_epi64(
            lanes[7], lanes[6], lanes[5], lanes[4],
            lanes[3], lanes[2], lanes[1], lanes[0]
        );
    }
}

static void bitplanes_to_bytes(__m512i* bitplanes, uint8_t* bytes, int count) {
    // Convert AVX-512 bit-planes back to byte array
    alignas(64) uint64_t lanes[8];

    for (int i = 0; i < count && i < 64; i++) {
        bytes[i] = 0;

        for (int bit = 0; bit < 8; bit++) {
            // Store __m512i to array
            _mm512_store_si512((__m512i*)lanes, bitplanes[bit]);

            // Extract lane and position
            int lane = i / 8;
            int pos = i % 8;

            // Get bit value
            if (lanes[lane] & (1ULL << (pos * 8))) {
                bytes[i] |= (1 << bit);
            }
        }
    }
}

int main() {
    printf("AVX-512 Tower Field S-box Verification\n");
    printf("======================================\n\n");

    __m512i in_planes[8], out_planes[8];
    uint8_t input[64], output[64];

    int errors = 0;
    int tests = 256;

    // Test all 256 S-box values in batches of 64
    for (int batch = 0; batch < 4; batch++) {
        // Prepare input batch
        for (int i = 0; i < 64; i++) {
            input[i] = batch * 64 + i;
        }

        // Convert to bit-planes
        bytes_to_bitplanes(input, in_planes, 64);

        // Run circuit
        circuit(in_planes, out_planes);

        // Convert back to bytes
        bitplanes_to_bytes(out_planes, output, 64);

        // Verify
        for (int i = 0; i < 64; i++) {
            int val = batch * 64 + i;
            if (val >= 256) break;

            uint8_t expected = AES_SBOX[val];
            if (output[i] != expected) {
                if (errors < 10) {
                    printf("  ERROR at 0x%02x: got 0x%02x, expected 0x%02x\n",
                           val, output[i], expected);
                }
                errors++;
            }
        }
    }

    printf("\n");
    if (errors == 0) {
        printf("✅ All %d S-box values correct!\n", tests);
        printf("\nAVX-512 tower field S-box is computing correctly.\n");
        return 0;
    } else {
        printf("❌ FAILED: %d errors out of %d\n", errors, tests);
        return 1;
    }
}
