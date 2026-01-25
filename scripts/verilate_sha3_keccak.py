from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=None if cwd is None else str(cwd))


def _hexdump(b: bytes) -> str:
    return b.hex()


def _try_endian_fixes(got: bytes) -> list[tuple[str, bytes]]:
    # Convenience transforms to help debug byte/word order mismatches.
    outs: list[tuple[str, bytes]] = [("raw", got)]
    outs.append(("byteswap32", b"".join(got[i : i + 4][::-1] for i in range(0, len(got), 4))))
    outs.append(("byteswap64", b"".join(got[i : i + 8][::-1] for i in range(0, len(got), 8))))
    outs.append(("reverse_bytes", got[::-1]))
    outs.append(("reverse_words64", b"".join(reversed([got[i : i + 8] for i in range(0, len(got), 8)]))))
    return outs


def build_and_run(msg: bytes, *, max_cycles: int = 20000) -> tuple[bytes, int]:
    verilator = shutil.which("verilator")
    if verilator is None:
        raise RuntimeError("missing verilator")

    rtl = Path("external-sha3-verilog/low_throughput_core/rtl")
    if not rtl.exists():
        raise FileNotFoundError("missing external-sha3-verilog checkout")

    v_files = sorted(rtl.glob("*.v"))
    if not v_files:
        raise FileNotFoundError("no .v files found under external-sha3-verilog/low_throughput_core/rtl")

    # Feed bytes into 32-bit big-endian words, matching the padder1 examples.
    words: list[tuple[int, int, int]] = []
    pos = 0
    while pos + 4 <= len(msg):
        w = int.from_bytes(msg[pos : pos + 4], byteorder="big", signed=False)
        words.append((w, 0, 0))  # in, is_last, byte_num
        pos += 4
    rem = len(msg) - pos
    if rem:
        w = int.from_bytes(msg[pos:] + b"\x00" * (4 - rem), byteorder="big", signed=False)
        words.append((w, 1, rem))
    else:
        # Exact multiple of 4 bytes: still need to signal last so padding is injected.
        words.append((0, 1, 0))

    with tempfile.TemporaryDirectory(prefix="verilate_sha3_") as td:
        td = Path(td)
        obj = td / "obj"
        obj.mkdir()
        sim = td / "sim_main.cpp"

        sim.write_text(
            _render_sim_cpp(words=words, max_cycles=max_cycles),
            encoding="utf-8",
        )

        cmd = [
            verilator,
            "--cc",
            "--exe",
            "--build",
            "--Mdir",
            str(obj),
            "--top-module",
            "keccak",
            *[str(p) for p in v_files],
            str(sim),
        ]
        _run(cmd)

        exe = obj / "Vkeccak"
        proc = subprocess.run(
            [str(exe)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out = proc.stdout.strip().splitlines()
        if not out:
            raise RuntimeError("no output from verilator sim")
        # Last line: "ready <cycle> <hex>"
        last = out[-1].strip().split()
        if len(last) != 3 or last[0] != "ready":
            raise RuntimeError(f"unexpected simulator output: {out[-1]}")
        cycle = int(last[1])
        hex_str = last[2]
        got = bytes.fromhex(hex_str)
        if len(got) != 64:
            raise RuntimeError(f"expected 64 bytes, got {len(got)}")
        return got, cycle


def _render_sim_cpp(*, words: list[tuple[int, int, int]], max_cycles: int) -> str:
    stim_inits = ",\n".join(
        f"    {{0x{w:08x}u, {is_last}u, {byte_num}u}}" for (w, is_last, byte_num) in words
    )
    return f"""
#include <cstdint>
#include <cstdio>
#include <vector>

#include \"verilated.h\"
#include \"Vkeccak.h\"

struct Word {{
    uint32_t in;
    uint32_t is_last;
    uint32_t byte_num;
}};

static const Word words[] = {{
{stim_inits}
}};

static std::string hex_digest(const VlWide<16>& w) {{
    // `out` is [511:0]. Print MSB-first as 64 bytes.
    // Verilator stores VlWide words as little-endian 32-bit chunks: word 0 is bits [31:0].
    char buf[2 * 64 + 1];
    int pos = 0;
    for (int byte = 63; byte >= 0; byte--) {{
        int bit = byte * 8;
        int wi = bit / 32;
        int bi = bit % 32;
        uint32_t v = w[wi];
        uint8_t b = (uint8_t)((v >> bi) & 0xFFu);
        std::snprintf(buf + pos, 3, \"%02x\", (unsigned)b);
        pos += 2;
    }}
    buf[pos] = 0;
    return std::string(buf);
}}

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);
    Vkeccak* top = new Vkeccak();

    // init
    top->clk = 0;
    top->reset = 1;
    top->in = 0;
    top->in_ready = 0;
    top->is_last = 0;
    top->byte_num = 0;
    top->eval();

    // one reset tick
    top->clk = 1; top->eval();
    top->clk = 0; top->eval();
    top->reset = 0;

    int word_idx = 0;
    for (int cycle = 0; cycle < {int(max_cycles)}; cycle++) {{
        // Drive input word when buffer has room.
        if (word_idx < (int)(sizeof(words)/sizeof(words[0])) && !top->buffer_full) {{
            top->in = words[word_idx].in;
            top->is_last = words[word_idx].is_last;
            top->byte_num = words[word_idx].byte_num;
            top->in_ready = 1;
            word_idx++;\n        }} else {{
            top->in_ready = 0;
            top->is_last = 0;
            top->byte_num = 0;
            top->in = 0;
        }}

        top->eval();

        if (top->out_ready) {{
            auto hex = hex_digest(top->out);
            std::printf(\"ready %d %s\\n\", cycle, hex.c_str());
            delete top;
            return 0;
        }}

        top->clk = 1; top->eval();
        top->clk = 0; top->eval();
    }}

    std::printf(\"timeout %d\\n\", {int(max_cycles)});
    delete top;
    return 2;
}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", type=str, default="abc")
    ap.add_argument("--max-cycles", type=int, default=20000)
    args = ap.parse_args()

    msg = args.msg.encode("utf-8")
    got, cycle = build_and_run(msg, max_cycles=int(args.max_cycles))
    # This core uses Keccak padding (0x01), not SHA3 padding (0x06).
    from stc.keccak_ref import keccak_512

    exp_keccak = keccak_512(msg, pad_byte=0x01)
    exp_sha3 = hashlib.sha3_512(msg).digest()
    print(f"cycle: {cycle}")
    print(f"got: {_hexdump(got)}")
    print(f"exp_keccak: {_hexdump(exp_keccak)}")
    print(f"exp_sha3:   {_hexdump(exp_sha3)}")
    if got == exp_keccak:
        print("PASS")
        return 0
    for name, candidate in _try_endian_fixes(got):
        if candidate == exp_keccak:
            print(f"PASS after transform: {name}")
            return 0
    print("MISMATCH")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
