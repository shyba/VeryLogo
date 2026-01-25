#!/usr/bin/env python3
"""
AES S-box via Tower Field Decomposition

This implements the AES S-box using tower field arithmetic:
  GF(2^8) = GF((2^4)^2) = GF(((2^2)^2)^2)

The S-box computes: S(x) = A * inv(x) + c
where inv(x) is the multiplicative inverse in GF(2^8), with inv(0) = 0.

Tower field inversion requires exactly 32 AND gates (multiplications),
with the rest being XOR gates (additions in GF(2)).

Reference: Boyar & Peralta, Canright's compact AES implementation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from stc.circuit_synth import CircuitState
from stc.bitslice import AES_SBOX_TABLE


@dataclass
class TowerFieldBuilder:
    """Builds a circuit using tower field arithmetic."""

    input_bits: int
    gates: list[tuple[str, int, int]]

    def __init__(self, input_bits: int = 8):
        self.input_bits = input_bits
        self.gates = []

    def input(self, i: int) -> int:
        """Get input signal index."""
        assert 0 <= i < self.input_bits
        return i

    def xor(self, a: int, b: int) -> int:
        """Add XOR gate, return output index."""
        idx = self.input_bits + len(self.gates)
        self.gates.append(("xor", a, b))
        return idx

    def and_(self, a: int, b: int) -> int:
        """Add AND gate, return output index."""
        idx = self.input_bits + len(self.gates)
        self.gates.append(("and", a, b))
        return idx

    def xor3(self, a: int, b: int, c: int) -> int:
        """XOR three values."""
        return self.xor(self.xor(a, b), c)

    def xor4(self, a: int, b: int, c: int, d: int) -> int:
        """XOR four values."""
        return self.xor(self.xor(a, b), self.xor(c, d))


def build_tower_field_sbox() -> CircuitState:
    """
    Build AES S-box circuit using tower field decomposition.

    The tower field approach decomposes GF(2^8) inversion as:
    1. Map input from polynomial basis to tower basis
    2. Compute inverse using tower field arithmetic (32 ANDs)
    3. Map back to polynomial basis
    4. Apply affine transformation

    This implementation follows the structure from Canright's paper
    and Boyar-Peralta's optimized circuit.
    """
    b = TowerFieldBuilder(8)

    # Input bits: x[0] is LSB, x[7] is MSB
    x = [b.input(i) for i in range(8)]

    # =========================================
    # Step 1: Change of basis (polynomial -> tower)
    # This is a linear transformation (XOR only)
    # =========================================

    # Top linear layer - compute intermediate signals
    # These are the "top" linear combinations from BP circuit
    t0 = b.xor(x[0], x[3])
    t1 = b.xor(x[0], x[5])
    t2 = b.xor(x[0], x[6])
    t3 = b.xor(x[3], x[5])
    t4 = b.xor(x[4], x[6])
    t5 = b.xor(t0, t4)
    t6 = b.xor(x[1], x[2])
    t7 = b.xor(x[7], t5)
    t8 = b.xor(x[7], t6)
    t9 = b.xor(t5, t6)
    t10 = b.xor(x[1], x[5])
    t11 = b.xor(x[2], x[5])
    t12 = b.xor(t2, t3)
    t13 = b.xor(t4, t10)
    t14 = b.xor(t1, t9)
    t15 = b.xor(x[4], t8)
    t16 = b.xor(t0, t11)
    t17 = b.xor(t3, t16)

    # =========================================
    # Step 2: Non-linear middle section (32 ANDs)
    # This computes the GF(2^4) inversion
    # =========================================

    # First set of AND gates (GF(2^2) multiplications)
    m0 = b.and_(t12, t5)
    m1 = b.and_(t3, t8)
    m2 = b.and_(t15, t2)
    m3 = b.and_(t6, t7)
    m4 = b.and_(t10, t14)
    m5 = b.and_(t4, t9)
    m6 = b.and_(t13, t17)
    m7 = b.and_(t16, t0)

    # XOR combinations of AND results
    m8 = b.xor(m0, m1)
    m9 = b.xor(m2, m3)
    m10 = b.xor(m4, m5)
    m11 = b.xor(m6, m7)
    m12 = b.xor(m8, m9)
    m13 = b.xor(m10, m11)
    m14 = b.xor(m12, m13)

    # Second set - more AND gates
    m15 = b.and_(m14, t12)
    m16 = b.and_(m14, t3)
    m17 = b.and_(m14, t15)
    m18 = b.and_(m14, t6)
    m19 = b.and_(m14, t10)
    m20 = b.and_(m14, t4)
    m21 = b.and_(m14, t13)
    m22 = b.and_(m14, t16)

    # More XOR combinations
    m23 = b.xor(m15, m0)
    m24 = b.xor(m16, m1)
    m25 = b.xor(m17, m2)
    m26 = b.xor(m18, m3)
    m27 = b.xor(m19, m4)
    m28 = b.xor(m20, m5)
    m29 = b.xor(m21, m6)
    m30 = b.xor(m22, m7)

    # Third set of AND gates
    m31 = b.and_(m23, t5)
    m32 = b.and_(m24, t8)
    m33 = b.and_(m25, t2)
    m34 = b.and_(m26, t7)
    m35 = b.and_(m27, t14)
    m36 = b.and_(m28, t9)
    m37 = b.and_(m29, t17)
    m38 = b.and_(m30, t0)

    # Fourth set of AND gates
    m39 = b.and_(m23, t7)
    m40 = b.and_(m24, t5)
    m41 = b.and_(m25, t8)
    m42 = b.and_(m26, t2)
    m43 = b.and_(m27, t9)
    m44 = b.and_(m28, t14)
    m45 = b.and_(m29, t0)
    m46 = b.and_(m30, t17)

    # =========================================
    # Step 3: Bottom linear layer (change basis back + affine)
    # =========================================

    # Combine results
    n0 = b.xor(m31, m39)
    n1 = b.xor(m32, m40)
    n2 = b.xor(m33, m41)
    n3 = b.xor(m34, m42)
    n4 = b.xor(m35, m43)
    n5 = b.xor(m36, m44)
    n6 = b.xor(m37, m45)
    n7 = b.xor(m38, m46)

    # More linear combinations for the affine part
    p0 = b.xor(n0, n1)
    p1 = b.xor(n2, n3)
    p2 = b.xor(n4, n5)
    p3 = b.xor(n6, n7)
    p4 = b.xor(p0, p1)
    p5 = b.xor(p2, p3)
    p6 = b.xor(p4, p5)

    # Final affine transformation to get S-box output
    # S-box affine: y = Ax + c where c = 0x63
    y0 = b.xor(n0, b.xor(p6, b.xor(n4, n6)))  # includes constant bit
    y1 = b.xor(n1, b.xor(p6, b.xor(n5, n7)))
    y2 = b.xor(n2, b.xor(p4, n6))
    y3 = b.xor(n3, b.xor(p4, n7))
    y4 = b.xor(n4, b.xor(p5, n0))
    y5 = b.xor(n5, b.xor(p5, b.xor(n1, p6)))
    y6 = b.xor(n6, b.xor(p0, b.xor(n2, p6)))
    y7 = b.xor(n7, b.xor(p1, n3))

    # Build outputs (with constant 0x63 = 01100011 added via inversions)
    # Affine constant bits: [1,1,0,0,0,1,1,0] for bits 0-7
    outputs = [
        (y0, True),  # bit 0: inverted (constant bit = 1)
        (y1, True),  # bit 1: inverted (constant bit = 1)
        (y2, False),  # bit 2: not inverted
        (y3, False),  # bit 3: not inverted
        (y4, False),  # bit 4: not inverted
        (y5, True),  # bit 5: inverted (constant bit = 1)
        (y6, True),  # bit 6: inverted (constant bit = 1)
        (y7, False),  # bit 7: not inverted
    ]

    return CircuitState(
        input_bits=8,
        output_bits=8,
        gates=b.gates,
        outputs=outputs,
        gate_count=len(b.gates),
    )


def build_canright_sbox() -> CircuitState:
    """
    Build AES S-box using Canright's optimized tower field construction.

    This is a cleaner implementation based on the actual tower field math:
    - GF(2^8) represented as GF((2^4)^2) with irreducible x^2 + x + N
    - GF(2^4) represented as GF((2^2)^2) with irreducible x^2 + x + n
    - GF(2^2) represented with irreducible x^2 + x + 1

    Inversion: (ah, al)^-1 where element is ah*x + al in GF(2^4) over GF(2^2)
    """
    b = TowerFieldBuilder(8)
    x = [b.input(i) for i in range(8)]

    # Change of basis matrix (polynomial to composite/tower)
    # From Canright's paper - maps standard AES polynomial basis to tower basis

    # Compute tower basis representation
    # a = ah || al where ah, al are 4-bit GF(2^4) elements
    # Each of those is bh || bl where bh, bl are 2-bit GF(2^2) elements

    # Linear transformation: polynomial -> tower
    # This specific matrix is from optimized implementations
    u0 = b.xor(x[0], x[6])
    u1 = b.xor(x[5], x[0])
    u2 = b.xor(x[1], x[2])
    u3 = b.xor(x[7], x[4])
    u4 = b.xor(u0, x[3])
    u5 = b.xor(u1, x[4])
    u6 = b.xor(u2, u3)
    u7 = b.xor(x[2], x[7])

    # Tower basis components (4 2-bit pairs)
    # ah = (a7,a6,a5,a4), al = (a3,a2,a1,a0) in tower
    a7 = b.xor(u4, u6)
    a6 = b.xor(x[7], b.xor(x[1], u4))
    a5 = b.xor(u1, u7)
    a4 = b.xor(u5, u2)
    a3 = b.xor(u0, x[7])
    a2 = b.xor(u3, u0)
    a1 = b.xor(x[4], u7)
    a0 = b.xor(u6, u5)

    # ---- Tower field inversion ----
    # Invert (ah, al) in GF((2^4)^2)
    # Formula: d = ah^2 * N + ah*al + al^2
    #          ah' = al * d^-1
    #          al' = (ah + al) * d^-1

    # GF(2^2) squaring is linear (free in hardware)
    # GF(2^2) multiplication requires 1 AND + some XORs
    # GF(2^4) multiplication requires 3 GF(2^2) mults = 3 ANDs
    # GF(2^4) inversion requires GF(2^2) inversion (linear) + mults

    # --- Square ah in GF(2^4) ---
    # ah = (a7,a6) || (a5,a4) as two GF(2^2) elements
    # ah^2 in GF(2^4) with irreducible x^2+x+N
    ah_h = b.xor(a7, a5)  # high part of ah in GF(2^2)
    ah_l = b.xor(a6, a4)  # low part

    # Squaring in GF(2^2): (b1,b0)^2 = (b0, b0^b1) with irred x^2+x+1
    # ah_h^2
    s0_h = b.xor(ah_h, ah_h)  # This is 0 actually (x XOR x = 0)
    # Let's do this properly - squaring maps (b1,b0) -> (b1^b0, b0)
    # We need to track individual bits

    # Let me restart with explicit bit tracking
    # ah = (a7, a6, a5, a4) in tower = ah_H * y + ah_L in GF(2^4)
    # where ah_H = (a7, a6), ah_L = (a5, a4) are GF(2^2) elements

    ah_H_1, ah_H_0 = a7, a6  # GF(2^2) element ah_H
    ah_L_1, ah_L_0 = a5, a4  # GF(2^2) element ah_L
    al_H_1, al_H_0 = a3, a2  # GF(2^2) element al_H
    al_L_1, al_L_0 = a1, a0  # GF(2^2) element al_L

    # Square ah in GF(2^4): ah^2 = (ah_H^2)*N + ah_L^2
    # where N is the GF(2^2) constant in irreducible poly
    # Using N = (1,0) i.e. the element 'x' in GF(2^2)

    # GF(2^2) square: (b1,b0)^2 = (b0, b1^b0) per Canright
    # ah_H^2:
    ahH_sq_1 = ah_H_0
    ahH_sq_0 = b.xor(ah_H_1, ah_H_0)
    # ah_L^2:
    ahL_sq_1 = ah_L_0
    ahL_sq_0 = b.xor(ah_L_1, ah_L_0)

    # Multiply ah_H^2 by N=(1,0): (b1,b0)*(1,0) = (b0, b1^b0)
    ahHsqN_1 = ahH_sq_0
    ahHsqN_0 = b.xor(ahH_sq_1, ahH_sq_0)

    # ah^2 = ahHsqN + ahL_sq (addition in GF(2^2))
    ah_sq_1 = b.xor(ahHsqN_1, ahL_sq_1)
    ah_sq_0 = b.xor(ahHsqN_0, ahL_sq_0)

    # Similarly square al
    alH_sq_1 = al_H_0
    alH_sq_0 = b.xor(al_H_1, al_H_0)
    alL_sq_1 = al_L_0
    alL_sq_0 = b.xor(al_L_1, al_L_0)
    alHsqN_1 = alH_sq_0
    alHsqN_0 = b.xor(alH_sq_1, alH_sq_0)
    al_sq_1 = b.xor(alHsqN_1, alL_sq_1)
    al_sq_0 = b.xor(alHsqN_0, alL_sq_0)

    # Now compute d = ah^2 * N + ah*al + al^2 in GF(2^4)
    # This requires GF(2^4) multiplication ah*al

    # GF(2^4) multiply: (aH, aL) * (bH, bL) =
    #   ((aH*bH + aL*bL)*N + aH*bL + aL*bH, (aH+aL)*(bH+bL) + aH*bH)
    # Wait, let me use the simpler form:
    # (aH*y + aL)(bH*y + bL) = aH*bH*y^2 + (aH*bL + aL*bH)*y + aL*bL
    # With y^2 = y + N: = aH*bH*(y+N) + (aH*bL + aL*bH)*y + aL*bL
    # = (aH*bH + aH*bL + aL*bH)*y + (aH*bH*N + aL*bL)
    # High part: aH*bH + aH*bL + aL*bH
    # Low part: aH*bH*N + aL*bL

    # For ah*al:
    # Need GF(2^2) multiplies: ah_H*al_H, ah_H*al_L, ah_L*al_H, ah_L*al_L

    # GF(2^2) multiply (a1,a0)*(b1,b0):
    # = a1*b1*x^2 + (a1*b0+a0*b1)*x + a0*b0
    # With x^2 = x+1: = a1*b1*(x+1) + (a1*b0+a0*b1)*x + a0*b0
    # = (a1*b1 + a1*b0 + a0*b1)*x + (a1*b1 + a0*b0)

    def gf4_mul(a1, a0, b1, b0):
        """Multiply two GF(2^2) elements, returns (r1, r0). Uses 1 AND."""
        # Karatsuba-like:
        # p = a1*b1, q = a0*b0, r = (a1^a0)*(b1^b0)
        # result_1 = p ^ r, result_0 = p ^ q
        p = b.and_(a1, b1)
        q = b.and_(a0, b0)
        r = b.and_(b.xor(a1, a0), b.xor(b1, b0))
        return b.xor(p, r), b.xor(p, q)

    # ah_H * al_H
    t1_1, t1_0 = gf4_mul(ah_H_1, ah_H_0, al_H_1, al_H_0)
    # ah_H * al_L
    t2_1, t2_0 = gf4_mul(ah_H_1, ah_H_0, al_L_1, al_L_0)
    # ah_L * al_H
    t3_1, t3_0 = gf4_mul(ah_L_1, ah_L_0, al_H_1, al_H_0)
    # ah_L * al_L
    t4_1, t4_0 = gf4_mul(ah_L_1, ah_L_0, al_L_1, al_L_0)

    # ah*al high part: t1 + t2 + t3
    ahal_H_1 = b.xor(t1_1, b.xor(t2_1, t3_1))
    ahal_H_0 = b.xor(t1_0, b.xor(t2_0, t3_0))
    # ah*al low part: t1*N + t4
    t1N_1 = t1_0
    t1N_0 = b.xor(t1_1, t1_0)
    ahal_L_1 = b.xor(t1N_1, t4_1)
    ahal_L_0 = b.xor(t1N_0, t4_0)

    # d = ah^2*N + ah*al + al^2
    # ah^2 is (ah_sq_1, ah_sq_0) - this is one GF(2^2) element
    # Need to treat it as GF(2^4)... let me reconsider

    # Actually ah^2 should be a GF(2^4) element (2 GF(2^2) parts)
    # Let me redo: ah = (ah_H, ah_L) as GF(2^4) = ah_H*y + ah_L
    # ah^2 = ah_H^2*y^2 + ah_L^2 = ah_H^2*(y+N) + ah_L^2
    #      = ah_H^2*y + (ah_H^2*N + ah_L^2)
    # So ah^2 as GF(2^4) = (ah_H^2, ah_H^2*N + ah_L^2)

    # ah_H^2 in GF(2^2):
    ah_H_sq_1 = ah_H_0
    ah_H_sq_0 = b.xor(ah_H_1, ah_H_0)
    # ah_L^2:
    ah_L_sq_1 = ah_L_0
    ah_L_sq_0 = b.xor(ah_L_1, ah_L_0)
    # ah_H^2 * N:
    ahHsq_N_1 = ah_H_sq_0
    ahHsq_N_0 = b.xor(ah_H_sq_1, ah_H_sq_0)

    # ah^2 as GF(2^4) element: high = ah_H^2, low = ah_H^2*N + ah_L^2
    ahsq_H_1 = ah_H_sq_1
    ahsq_H_0 = ah_H_sq_0
    ahsq_L_1 = b.xor(ahHsq_N_1, ah_L_sq_1)
    ahsq_L_0 = b.xor(ahHsq_N_0, ah_L_sq_0)

    # al^2 as GF(2^4):
    al_H_sq_1 = al_H_0
    al_H_sq_0 = b.xor(al_H_1, al_H_0)
    al_L_sq_1 = al_L_0
    al_L_sq_0 = b.xor(al_L_1, al_L_0)
    alHsq_N_1 = al_H_sq_0
    alHsq_N_0 = b.xor(al_H_sq_1, al_H_sq_0)
    alsq_H_1 = al_H_sq_1
    alsq_H_0 = al_H_sq_0
    alsq_L_1 = b.xor(alHsq_N_1, al_L_sq_1)
    alsq_L_0 = b.xor(alHsq_N_0, al_L_sq_0)

    # ah^2 * N (multiply GF(2^4) by N which is (0,N_gf22) = (0, (1,0)))
    # (H, L) * (0, N) = (H*N, L*N) -- simplified since high part of N is 0
    # Wait, N in GF(2^4) is just the constant N from irreducible
    # For x^2 + x + N with N = GF(2^2) element (1,0)
    # Multiplying by N: (aH, aL) * N = (0, aH*N + aL*N) -- no that's wrong
    # Scalar mult: (aH, aL) * c = (aH*c, aL*c) where c is GF(2^2) element

    # ahsq * N (scalar multiply by N=(1,0)):
    ahsqN_H_1 = ahsq_H_0  # ahsq_H * N
    ahsqN_H_0 = b.xor(ahsq_H_1, ahsq_H_0)
    ahsqN_L_1 = ahsq_L_0  # ahsq_L * N
    ahsqN_L_0 = b.xor(ahsq_L_1, ahsq_L_0)

    # d = ah^2*N + ah*al + al^2 (all GF(2^4) additions = XOR componentwise)
    d_H_1 = b.xor(ahsqN_H_1, b.xor(ahal_H_1, alsq_H_1))
    d_H_0 = b.xor(ahsqN_H_0, b.xor(ahal_H_0, alsq_H_0))
    d_L_1 = b.xor(ahsqN_L_1, b.xor(ahal_L_1, alsq_L_1))
    d_L_0 = b.xor(ahsqN_L_0, b.xor(ahal_L_0, alsq_L_0))

    # Now invert d in GF(2^4)
    # d^-1 where d = (d_H, d_L) in GF(2^4)
    # Use same formula: e = d_H^2*n + d_H*d_L + d_L^2 in GF(2^2)
    # d_H^-1_candidate = d_L * e^-1
    # d_L^-1_candidate = (d_H + d_L) * e^-1

    # d_H^2 in GF(2^2):
    dH_sq_1 = d_H_0
    dH_sq_0 = b.xor(d_H_1, d_H_0)
    # d_L^2:
    dL_sq_1 = d_L_0
    dL_sq_0 = b.xor(d_L_1, d_L_0)

    # d_H * d_L:
    dHdL_1, dHdL_0 = gf4_mul(d_H_1, d_H_0, d_L_1, d_L_0)

    # d_H^2 * n where n=(1,0) for GF(2^2) over GF(2):
    dHsq_n_1 = dH_sq_0
    dHsq_n_0 = b.xor(dH_sq_1, dH_sq_0)

    # e = d_H^2*n + d_H*d_L + d_L^2
    e_1 = b.xor(dHsq_n_1, b.xor(dHdL_1, dL_sq_1))
    e_0 = b.xor(dHsq_n_0, b.xor(dHdL_0, dL_sq_0))

    # Invert e in GF(2^2): e^-1
    # For GF(2^2) with irred x^2+x+1:
    # (e1, e0)^-1 = (e1, e1+e0) when e != 0
    # This is because in GF(4), inv table is: 1->1, x->x+1, x+1->x
    # Actually let me verify: elements are 0,1,x,x+1
    # 1*1=1, x*(x+1) = x^2+x = 1 (using x^2=x+1), (x+1)*x = 1
    # So: 1^-1=1=(0,1), x^-1=x+1=(1,1), (x+1)^-1=x=(1,0)
    # (0,1)^-1 = (0,1)
    # (1,0)^-1 = (1,1)
    # (1,1)^-1 = (1,0)
    # Formula: (a,b)^-1 = (a, a+b) -- let's check
    # (0,1) -> (0, 0+1) = (0,1) ✓
    # (1,0) -> (1, 1+0) = (1,1) ✓
    # (1,1) -> (1, 1+1) = (1,0) ✓
    e_inv_1 = e_1
    e_inv_0 = b.xor(e_1, e_0)

    # d^-1 in GF(2^4):
    # d_inv_H = d_L * e^-1
    # d_inv_L = (d_H + d_L) * e^-1
    d_inv_H_1, d_inv_H_0 = gf4_mul(d_L_1, d_L_0, e_inv_1, e_inv_0)
    dH_plus_dL_1 = b.xor(d_H_1, d_L_1)
    dH_plus_dL_0 = b.xor(d_H_0, d_L_0)
    d_inv_L_1, d_inv_L_0 = gf4_mul(dH_plus_dL_1, dH_plus_dL_0, e_inv_1, e_inv_0)

    # Now compute (ah, al)^-1 in GF(2^8):
    # ah' = al * d^-1
    # al' = (ah + al) * d^-1

    # al * d^-1 (GF(2^4) multiply)
    # al = (al_H, al_L), d^-1 = (d_inv_H, d_inv_L)

    # GF(2^4) multiply needs 4 GF(2^2) multiplies...
    # Let me use the Karatsuba form to minimize ANDs
    # (aH, aL) * (bH, bL):
    # p1 = aH * bH
    # p2 = aL * bL
    # p3 = (aH + aL) * (bH + bL)
    # result_H = p1 + p3 (which is p1 + aH*bL + aL*bH)
    # result_L = p1*N + p2

    def gf16_mul(aH1, aH0, aL1, aL0, bH1, bH0, bL1, bL0):
        """Multiply two GF(2^4) elements, returns ((rH1,rH0), (rL1,rL0)). Uses 3 ANDs."""
        p1_1, p1_0 = gf4_mul(aH1, aH0, bH1, bH0)
        p2_1, p2_0 = gf4_mul(aL1, aL0, bL1, bL0)
        aHL_1 = b.xor(aH1, aL1)
        aHL_0 = b.xor(aH0, aL0)
        bHL_1 = b.xor(bH1, bL1)
        bHL_0 = b.xor(bH0, bL0)
        p3_1, p3_0 = gf4_mul(aHL_1, aHL_0, bHL_1, bHL_0)
        # result_H = p1 + p3 + p2 = p3 + p1 + p2 (since we want aH*bL + aL*bH)
        # Actually: p3 = (aH+aL)*(bH+bL) = aH*bH + aH*bL + aL*bH + aL*bL = p1 + aH*bL + aL*bH + p2
        # So aH*bL + aL*bH = p3 + p1 + p2
        # result_H = p1 + (p3 + p1 + p2) = p3 + p2
        rH_1 = b.xor(p3_1, p2_1)
        rH_0 = b.xor(p3_0, p2_0)
        # result_L = p1*N + p2
        p1N_1 = p1_0
        p1N_0 = b.xor(p1_1, p1_0)
        rL_1 = b.xor(p1N_1, p2_1)
        rL_0 = b.xor(p1N_0, p2_0)
        return rH_1, rH_0, rL_1, rL_0

    # ah' = al * d^-1
    ah_p_H_1, ah_p_H_0, ah_p_L_1, ah_p_L_0 = gf16_mul(
        al_H_1, al_H_0, al_L_1, al_L_0, d_inv_H_1, d_inv_H_0, d_inv_L_1, d_inv_L_0
    )

    # al' = (ah + al) * d^-1
    ah_al_H_1 = b.xor(ah_H_1, al_H_1)
    ah_al_H_0 = b.xor(ah_H_0, al_H_0)
    ah_al_L_1 = b.xor(ah_L_1, al_L_1)
    ah_al_L_0 = b.xor(ah_L_0, al_L_0)

    al_p_H_1, al_p_H_0, al_p_L_1, al_p_L_0 = gf16_mul(
        ah_al_H_1,
        ah_al_H_0,
        ah_al_L_1,
        ah_al_L_0,
        d_inv_H_1,
        d_inv_H_0,
        d_inv_L_1,
        d_inv_L_0,
    )

    # Now we have the inverse in tower basis: (ah', al')
    # Map back to polynomial basis and apply affine transform

    # Inverse tower basis bits:
    inv = [
        al_p_L_0,
        al_p_L_1,
        al_p_H_0,
        al_p_H_1,
        ah_p_L_0,
        ah_p_L_1,
        ah_p_H_0,
        ah_p_H_1,
    ]

    # Change of basis: tower -> polynomial
    # Then apply AES affine transformation
    # Combined matrix (from Canright):

    v0 = b.xor(inv[0], inv[1])
    v1 = b.xor(inv[2], inv[3])
    v2 = b.xor(inv[4], inv[5])
    v3 = b.xor(inv[6], inv[7])
    v4 = b.xor(v0, v1)
    v5 = b.xor(v2, v3)
    v6 = b.xor(v4, v5)

    # Final output combining basis change + affine
    # The exact matrix depends on the specific basis chosen
    # For now, let's use a simplified approach

    y0 = b.xor(inv[0], b.xor(inv[2], b.xor(inv[4], v5)))
    y1 = b.xor(inv[1], b.xor(inv[3], b.xor(inv[5], v4)))
    y2 = b.xor(inv[2], b.xor(inv[4], v3))
    y3 = b.xor(inv[3], b.xor(inv[5], v2))
    y4 = b.xor(inv[4], b.xor(inv[6], v1))
    y5 = b.xor(inv[5], b.xor(inv[7], v0))
    y6 = b.xor(inv[6], b.xor(inv[0], v5))
    y7 = b.xor(inv[7], b.xor(inv[1], v4))

    # Output with affine constant 0x63 = 01100011
    outputs = [
        (y0, True),  # constant bit 1
        (y1, True),  # constant bit 1
        (y2, False),
        (y3, False),
        (y4, False),
        (y5, True),  # constant bit 1
        (y6, True),  # constant bit 1
        (y7, False),
    ]

    return CircuitState(
        input_bits=8,
        output_bits=8,
        gates=b.gates,
        outputs=outputs,
        gate_count=len(b.gates),
    )


def verify_sbox(circuit: CircuitState, name: str) -> bool:
    """Verify circuit against AES S-box table."""
    errors = 0
    for i in range(256):
        result = circuit.evaluate(i)
        expected = AES_SBOX_TABLE[i]
        if result != expected:
            if errors < 5:
                print(
                    f"  {name} error at {i}: got {result:#04x}, expected {expected:#04x}"
                )
            errors += 1
    if errors:
        print(f"  {name}: {errors} errors out of 256")
        return False
    return True


def main():
    print("Building tower field S-box circuits...")
    print()

    # Try the first implementation
    print("Method 1: Direct tower field construction")
    circuit1 = build_tower_field_sbox()
    print(
        f"  Gates: {circuit1.gate_count} ({circuit1.and_count} AND, {circuit1.xor_count} XOR)"
    )
    correct1 = verify_sbox(circuit1, "Method 1")
    print(f"  Correct: {correct1}")
    print()

    # Try Canright's method
    print("Method 2: Canright's tower field")
    circuit2 = build_canright_sbox()
    print(
        f"  Gates: {circuit2.gate_count} ({circuit2.and_count} AND, {circuit2.xor_count} XOR)"
    )
    correct2 = verify_sbox(circuit2, "Method 2")
    print(f"  Correct: {correct2}")
    print()

    # If we have a correct circuit, try optimizing it
    if correct1:
        print("Optimizing Method 1 circuit...")
        opt = circuit1
        opt = opt.eliminate_common_subexpressions()
        opt = opt.apply_algebraic_rewrites()
        opt = opt.eliminate_dead_code()
        opt = opt.flatten_xor_trees()
        print(
            f"  After Phase 1: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)"
        )

        # Try BP linear optimization
        opt = opt.optimize_linear_layers()
        print(
            f"  After BP: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)"
        )

        # Verify still correct
        if verify_sbox(opt, "Optimized"):
            print("  Still correct after optimization!")

    if correct2:
        print("Optimizing Method 2 circuit...")
        opt = circuit2
        opt = opt.eliminate_common_subexpressions()
        opt = opt.apply_algebraic_rewrites()
        opt = opt.eliminate_dead_code()
        opt = opt.flatten_xor_trees()
        print(
            f"  After Phase 1: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)"
        )

        opt = opt.optimize_linear_layers()
        print(
            f"  After BP: {opt.gate_count} gates ({opt.and_count} AND, {opt.xor_count} XOR)"
        )

        if verify_sbox(opt, "Optimized"):
            print("  Still correct after optimization!")


if __name__ == "__main__":
    main()
