from __future__ import annotations

from stc.io_map import IoMap, validate_io_map
from stc.tick_ir import TickIR


def emit_avr_project(ir: TickIR, *, io_map: IoMap) -> dict[str, str]:
    validate_io_map(ir, io_map)

    out_mask = 0
    for spec in io_map.outputs.values():
        lsb = int(spec["lsb"])
        w = int(spec["width"])
        out_mask |= ((1 << w) - 1) << lsb

    makefile = "\n".join(
        [
            "MCU=attiny85",
            "F_CPU=8000000",
            "CC=avr-gcc",
            "OBJCOPY=avr-objcopy",
            "CFLAGS=-std=c99 -Os -Wall -Wextra -mmcu=$(MCU) -DF_CPU=$(F_CPU)UL",
            "",
            "all: main.hex",
            "",
            "main.elf: main.c avr.c",
            "\t$(CC) $(CFLAGS) -o $@ $^",
            "",
            "main.hex: main.elf",
            "\t$(OBJCOPY) -O ihex -R .eeprom $< $@",
            "",
            "clean:",
            "\trm -f main.elf main.hex",
            "",
        ]
    )

    main_c = "\n".join(
        [
            "#include <avr/io.h>",
            "#include <stdint.h>",
            "",
            "void stc_reset(void);",
            "void stc_tick(void);",
            "",
            "int main(void) {",
            f"  DDRB = (uint8_t)((DDRB & (uint8_t)~{hex(out_mask)}u) | {hex(out_mask)}u);",
            "  stc_reset();",
            "  for (;;) {",
            "    stc_tick();",
            "  }",
            "}",
            "",
        ]
    )

    return {"Makefile": makefile, "main.c": main_c}
