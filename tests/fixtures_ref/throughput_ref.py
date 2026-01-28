def vec_add(a, b, N=1024):
    result = []
    for i in range(N):
        a_lane = (a >> (32 * i)) & 0xFFFFFFFF
        b_lane = (b >> (32 * i)) & 0xFFFFFFFF
        c_lane = (a_lane + b_lane) & 0xFFFFFFFF
        result.append(c_lane)

    output = 0
    for i in range(N):
        output |= result[i] << (32 * i)
    return output


def xor_tree(inputs, depth=24):
    if not isinstance(inputs, list):
        if depth == 0:
            return inputs
        left_input = inputs & ((1 << (inputs.bit_length() // 2)) - 1)
        right_input = inputs >> (inputs.bit_length() // 2)
        return xor_tree(left_input, depth - 1) ^ xor_tree(right_input, depth - 1)

    if len(inputs) == 1:
        return inputs[0]

    if depth == 0:
        result = 0
        for val in inputs:
            result ^= val
        return result

    mid = len(inputs) // 2
    left = xor_tree(inputs[:mid], depth - 1)
    right = xor_tree(inputs[mid:], depth - 1)
    return left ^ right
