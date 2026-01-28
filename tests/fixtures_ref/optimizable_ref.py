def roundN(in_val, R=8, ticks=None, rst_at=None):
    if ticks is None:
        ticks = R + 1

    s = 0
    r = 0
    out = 0

    states = []

    def f(x, k):
        x = x & 0xFFFFFFFFFFFFFFFF
        k = k & 0xFF
        rotated = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
        added = (x + k) & 0xFFFFFFFFFFFFFFFF
        result = rotated ^ added
        return result & 0xFFFFFFFFFFFFFFFF

    for tick in range(ticks):
        if rst_at is not None and tick in rst_at:
            s = in_val & 0xFFFFFFFFFFFFFFFF
            r = 0
        elif r == 0 and tick == 0:
            s = in_val & 0xFFFFFFFFFFFFFFFF
            r = 0
        elif r < R:
            s = f(s, r)
            r = (r + 1) & 0xFF

        out = s

        states.append({"s": s, "r": r, "out": out})

    return states


def nonlinear_island(x, y, width=64):
    x = x & ((1 << width) - 1)
    y = y & ((1 << width) - 1)

    core = (x & y) ^ ((x | y) & ((x ^ y) << 1))
    core = core & ((1 << width) - 1)

    r1 = ((core << 13) | (core >> (width - 13))) & ((1 << width) - 1)
    r2 = ((core << 37) | (core >> (width - 37))) & ((1 << width) - 1)
    r3 = ((core << 7) | (core >> (width - 7))) & ((1 << width) - 1)

    mixed = core ^ r1 ^ r2 ^ r3 ^ x ^ y

    final_rot = ((mixed << 23) | (mixed >> (width - 23))) & ((1 << width) - 1)
    result = final_rot ^ ((mixed >> 3) & ((1 << width) - 1))

    return result & ((1 << width) - 1)
