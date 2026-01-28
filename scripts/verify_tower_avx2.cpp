/**
 * Verify AVX2 tower field S-box against AES truth table.
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "aes_sbox_table.h"

// Forward declaration
void circuit(__m256i* in, __m256i* out);

static void bytes_to_bitplanes_avx2(const uint8_t* bytes, __m256i* bitplanes, int count) {
    // Convert byte array to AVX2 bit-planes (32 values per plane)
    for (int bit = 0; bit < 8; bit++) {
        uint32_t lanes[8] = {0};

        for (int i = 0; i < count && i < 32; i++) {
            if (bytes[i] & (1 << bit)) {
                int lane = i / 4;
                int pos = i % 4;
                lanes[lane] |= (1U << (pos * 8));
            }
        }

        bitplanes[bit] = _mm256_set_epi32(
            lanes[7], lanes[6], lanes[5], lanes[4],
            lanes[3], lanes[2], lanes[1], lanes[0]
        );
    }
}

static void bitplanes_to_bytes_avx2(__m256i* bitplanes, uint8_t* bytes, int count) {
    // Convert AVX2 bit-planes back to byte array
    alignas(32) uint32_t lanes[8];

    for (int i = 0; i < count && i < 32; i++) {
        bytes[i] = 0;

        for (int bit = 0; bit < 8; bit++) {
            // Store __m256i to array
            _mm256_store_si256((__m256i*)lanes, bitplanes[bit]);

            // Extract lane and position
            int lane = i / 4;
            int pos = i % 4;

            // Get bit value
            if (lanes[lane] & (1U << (pos * 8))) {
                bytes[i] |= (1 << bit);
            }
        }
    }
}

int main() {
    printf("AVX2 Tower Field S-box Verification\n");
    printf("===================================\n\n");

    __m256i in_planes[8], out_planes[8];
    uint8_t input[32], output[32];

    int errors = 0;
    int tests = 256;

    // Test all 256 S-box values in batches of 32
    for (int batch = 0; batch < 8; batch++) {
        // Prepare input batch
        for (int i = 0; i < 32; i++) {
            input[i] = batch * 32 + i;
        }

        // Convert to bit-planes
        bytes_to_bitplanes_avx2(input, in_planes, 32);

        // Run circuit
        circuit(in_planes, out_planes);

        // Convert back to bytes
        bitplanes_to_bytes_avx2(out_planes, output, 32);

        // Verify
        for (int i = 0; i < 32; i++) {
            int val = batch * 32 + i;
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
        printf("\nAVX2 tower field S-box is computing correctly.\n");
        return 0;
    } else {
        printf("❌ FAILED: %d errors out of %d\n", errors, tests);
        return 1;
    }
}
