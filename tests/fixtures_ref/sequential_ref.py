def acc_fsm(limit, x, ticks, rst_at=None):
    acc = 0
    i = 0

    states = []

    for tick in range(ticks):
        if rst_at is not None and tick in rst_at:
            acc = 0
            i = 0
        elif i < limit:
            acc = (acc + (x ^ i)) & 0xFFFFFFFF
            i = (i + 1) & 0xFFFFFFFF

        states.append({"acc": acc, "i": i})

    return states


def lfsr_next(state, taps=None):
    if taps is None:
        taps = [32, 22, 2, 1]

    bit = 0
    for tap in taps:
        bit ^= (state >> (tap - 1)) & 1

    next_state = ((state << 1) | bit) & 0xFFFFFFFF
    return next_state


def lfsr_table(seed, table, ticks, table_width=32, rst_at=None):
    if table_width not in [16, 32, 64]:
        raise ValueError("table_width must be 16, 32, or 64")

    if len(table) != (1 << table_width.bit_length() - 1):
        if table_width == 16:
            expected = 16
        elif table_width == 32:
            expected = 32
        elif table_width == 64:
            expected = 64
        if len(table) < expected:
            table = table + [0] * (expected - len(table))

    lfsr = seed & 0xFFFFFFFF
    acc = 0

    states = []

    for tick in range(ticks):
        if rst_at is not None and tick in rst_at:
            lfsr = seed & 0xFFFFFFFF
            acc = 0
        else:
            idx = lfsr & ((1 << table_width.bit_length() - 1) - 1)
            if idx < len(table):
                acc = (acc ^ table[idx]) & 0xFFFFFFFF
            lfsr = lfsr_next(lfsr)

        states.append({"lfsr": lfsr, "acc": acc})

    return states
