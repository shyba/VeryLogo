def rotmix(x, width=64):
    mask = (1 << width) - 1
    x = x & mask

    r1 = ((x << 3) | (x >> (width - 3))) & mask
    r2 = ((x << 51) | (x >> (width - 51))) & mask

    const = 0x9E3779B97F4A7C15 & mask
    added = (x + const) & mask

    y = (r1 ^ r2 ^ added) & mask
    return y


def slice_concat(x, width=64):
    mask = (1 << width) - 1
    x = x & mask

    hi = (x >> 32) & 0xFFFFFFFF
    lo = x & 0xFFFFFFFF

    q0 = lo & 0xFFFF
    q1 = (lo >> 16) & 0xFFFF
    q2 = hi & 0xFFFF
    q3 = (hi >> 16) & 0xFFFF

    permuted = (q1 << 48) | (q3 << 32) | (q0 << 16) | q2
    permuted = permuted & mask

    slice_a = (permuted >> 24) & 0xFFFFFF
    slice_b = permuted & 0xFFFFFF
    slice_c = (permuted >> 48) & 0xFFFF

    recon = (slice_c << 48) | (slice_a << 24) | slice_b
    recon = recon & mask

    odd_slice_0 = recon & 0x3FF
    odd_slice_1 = (recon >> 13) & 0x3FF
    odd_slice_2 = (recon >> 26) & 0x3FF
    odd_slice_3 = (recon >> 39) & 0x3FF

    top_bits = (recon >> 49) & 0x7FFF
    mid_bits_1 = (recon >> 10) & 0x7
    mid_bits_2 = (recon >> 23) & 0x7
    mid_bits_3 = (recon >> 36) & 0x7
    odd_slice_4 = (top_bits << 9) | (mid_bits_1 << 6) | (mid_bits_2 << 3) | mid_bits_3
    odd_slice_4 = odd_slice_4 & 0xFFFFFF

    y = (
        (odd_slice_3 << 54)
        | (odd_slice_4 << 30)
        | (odd_slice_2 << 20)
        | (odd_slice_1 << 10)
        | odd_slice_0
    )
    return y & mask
