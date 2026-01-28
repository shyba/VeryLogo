def pmux16(sel, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15):
    inputs = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]
    sel = sel & 0xF
    return inputs[sel] & 0xFFFFFFFFFFFFFFFF


def pmux32(
    sel,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    a16,
    a17,
    a18,
    a19,
    a20,
    a21,
    a22,
    a23,
    a24,
    a25,
    a26,
    a27,
    a28,
    a29,
    a30,
    a31,
):
    inputs = [
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
        a8,
        a9,
        a10,
        a11,
        a12,
        a13,
        a14,
        a15,
        a16,
        a17,
        a18,
        a19,
        a20,
        a21,
        a22,
        a23,
        a24,
        a25,
        a26,
        a27,
        a28,
        a29,
        a30,
        a31,
    ]
    sel = sel & 0x1F
    return inputs[sel] & 0xFFFFFFFFFFFFFFFF


def mux_reconverge(sel0, sel1, a, b, c, d, width=64):
    mask = (1 << width) - 1
    a = a & mask
    b = b & mask
    c = c & mask
    d = d & mask

    if sel0 == 0:
        m0 = a
    elif sel0 == 1:
        m0 = b
    elif sel0 == 2:
        m0 = c
    else:
        m0 = d

    if sel1 == 0:
        m1 = a
    elif sel1 == 1:
        m1 = b
    elif sel1 == 2:
        m1 = c
    else:
        m1 = d

    m2 = a if (sel0 == sel1) else b

    m3 = m0 if ((sel0 & 1) == (sel1 & 1)) else m1

    combined = (m0 ^ m1) & mask
    combined2 = (m2 + m3) & mask

    result = (combined ^ combined2) & mask

    return result
