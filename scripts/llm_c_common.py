"""Shared C template fragments for the LLM benchmark scripts.

The SIMD helper semantics (f2b/b2f, exp_ps, layernorm, gelu, softmax,
split_qkv, the pack functions, and the gemm driver) are property-tested in
tests/test_properties_simd.py and tests/test_properties_gemm.py; keep them
in sync with the property tests when changed. The bf16 8x32 asm kernel is
emitted by stc.gemm_asm and assembled separately.
"""

COMMON_C = r"""static inline uint16_t f2b(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    return (uint16_t)r;
}
static inline float b2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline __m512 exp_ps(__m512 x) {
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f));
    t = _mm512_min_ps(_mm512_max_ps(t, _mm512_set1_ps(-126.0f)), _mm512_set1_ps(127.0f));
    __m512i n = _mm512_cvtps_epi32(_mm512_roundscale_ps(t, 0));
    __m512 r = _mm512_sub_ps(t, _mm512_cvtepi32_ps(n));
    __m512 p = _mm512_set1_ps(0.001333355815f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.009618129108f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.055504108665f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.240226506959f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.693147180560f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.000000000000f));
    __m512i e = _mm512_slli_epi32(_mm512_add_epi32(n, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
}
/* pack B (KxN, row-major) into K-major/N-interleaved layout */
static void pack_b_bf16_32(const uint16_t* B, int32_t* Bp, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < 32; j++) {
                uint32_t v = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    v |= (uint32_t)B[(2*c + t)*N + col] << (16*t);
                Bp[((n0/32)*(K/2) + c)*32 + j] = v;
            }
}
/* pack B^T (B is N x K row-major) into the K-major/N-interleaved layout */
static void pack_b_bf16_32_T(const uint16_t* B, int32_t* Bp, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int c = 0; c < K/2; c++)
            for (int j = 0; j < 32; j++) {
                uint32_t v = 0; int col = n0 + j;
                for (int t = 0; t < 2; t++)
                    v |= (uint32_t)B[col*K + (2*c + t)] << (16*t);
                Bp[((n0/32)*(K/2) + c)*32 + j] = v;
            }
}
/* C(MxN) = A(MxK) . Bp; tiles 8x32; M%8==0, N%32==0, K%2==0 */
static void gemm(const uint16_t* A, const int32_t* Bp, float* C, int M, int N, int K) {
    for (int n0 = 0; n0 < N; n0 += 32)
        for (int m0 = 0; m0 < M; m0 += 8)
            gemm_bf16_8x32_asm(A + m0*K, Bp + (n0/32)*(K/2)*32, C + m0*N + n0, K, N);
}
static void to_bf16(const float* x, uint16_t* y, int n) {
    for (int i = 0; i < n; i++) y[i] = f2b(x[i]);
}
/* qkv (S x 3D) -> q,k,v (S x D each); columns are strided in the fused output */
static void split_qkv(const float* qkv, float* q, float* k, float* v, int S, int D) {
    for (int r = 0; r < S; r++) {
        memcpy(q + r*D, qkv + r*3*D, (size_t)D*4);
        memcpy(k + r*D, qkv + r*3*D + D, (size_t)D*4);
        memcpy(v + r*D, qkv + r*3*D + 2*D, (size_t)D*4);
    }
}
static void layernorm(float* x, int S, int D) {
    const int V = D / 16;
    for (int i = 0; i < S; i++) {
        float* r = x + i*D;
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) sum = _mm512_add_ps(sum, _mm512_loadu_ps(r + 16*j));
        float mean = _mm512_reduce_add_ps(sum) / (float)D;
        __m512 mvec = _mm512_set1_ps(mean);
        __m512 ss = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) {
            __m512 t = _mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec);
            ss = _mm512_fmadd_ps(t, t, ss);
        }
        float var = _mm512_reduce_add_ps(ss) / (float)D;
        __m512 iv = _mm512_set1_ps(1.0f / sqrtf(var + 1e-5f));
        for (int j = 0; j < V; j++)
            _mm512_storeu_ps(r + 16*j, _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec), iv));
    }
}
static void gelu(float* x, int n) {
    const float c = 0.7978845608f;
    for (int i = 0; i < n/16; i++) {
        __m512 v = _mm512_loadu_ps(x + 16*i);
        __m512 v3 = _mm512_mul_ps(v, _mm512_mul_ps(v, v));
        __m512 z = _mm512_mul_ps(_mm512_add_ps(v, _mm512_mul_ps(_mm512_set1_ps(0.044715f), v3)), _mm512_set1_ps(c));
        __m512 e = exp_ps(_mm512_add_ps(z, z));
        __m512 t = _mm512_div_ps(_mm512_sub_ps(e, _mm512_set1_ps(1.0f)), _mm512_add_ps(e, _mm512_set1_ps(1.0f)));
        __m512 g = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(v, _mm512_add_ps(_mm512_set1_ps(1.0f), t)));
        _mm512_storeu_ps(x + 16*i, g);
    }
}
static void softmax(float* x, int S) {
    const int V = S / 16;
    for (int i = 0; i < S; i++) {
        float* r = x + i*S;
        __m512 mx = _mm512_loadu_ps(r);
        for (int j = 1; j < V; j++)
            mx = _mm512_max_ps(mx, _mm512_loadu_ps(r + 16*j));
        float m = _mm512_reduce_max_ps(mx);
        __m512 mvec = _mm512_set1_ps(m);
        __m512 sum = _mm512_setzero_ps();
        for (int j = 0; j < V; j++) {
            __m512 e = exp_ps(_mm512_sub_ps(_mm512_loadu_ps(r + 16*j), mvec));
            _mm512_storeu_ps(r + 16*j, e);
            sum = _mm512_add_ps(sum, e);
        }
        float inv = 1.0f / _mm512_reduce_add_ps(sum);
        __m512 iv = _mm512_set1_ps(inv);
        for (int j = 0; j < V; j++)
            _mm512_storeu_ps(r + 16*j, _mm512_mul_ps(_mm512_loadu_ps(r + 16*j), iv));
    }
}
"""
