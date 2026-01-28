"""
SPIR-V emitter for VeryLogo MIR.

Generates SPIR-V assembly for Vulkan compute shaders.
"""

from __future__ import annotations

from stc.mir import (
    Binary,
    Unary,
    Ternary,
    Mux,
    Copy,
    Const,
    Load,
    Store,
    MIRFunction,
    VReg,
    PReg,
)
from stc.sched.regalloc import RegAllocation


class SPIRVEmitter:
    """Emit SPIR-V assembly from MIR."""

    def __init__(self):
        self.id_counter = 1
        self.type_ids = {}
        self.const_ids = {}
        self.reg_ids = {}
        self.lines = []

    def fresh_id(self) -> int:
        """Generate a fresh SPIR-V ID."""
        result = self.id_counter
        self.id_counter += 1
        return result

    def get_type_id(self, ty: str) -> int:
        """Get or create type ID."""
        if ty not in self.type_ids:
            self.type_ids[ty] = self.fresh_id()
        return self.type_ids[ty]

    def get_reg_id(self, reg) -> int:
        """Get or create register ID."""
        key = f"v{reg.id}" if isinstance(reg, VReg) else reg.name
        if key not in self.reg_ids:
            self.reg_ids[key] = self.fresh_id()
        return self.reg_ids[key]

    def emit_header(self, num_regs: int) -> None:
        """Emit SPIR-V header and declarations."""
        # Magic number, version 1.0, generator=0, bound (will update), schema=0
        bound = 1000  # Placeholder, will be updated
        self.lines.append(f"; SPIR-V")
        self.lines.append(f"; Version: 1.0")
        self.lines.append(f"; Generator: VeryLogo")
        self.lines.append(f"; Bound: {bound}")
        self.lines.append(f"; Schema: 0")
        self.lines.append("")

        # Capabilities
        self.lines.append("OpCapability Shader")
        self.lines.append("")

        # Extensions for StorageBuffer
        self.lines.append('OpExtension "SPV_KHR_storage_buffer_storage_class"')
        self.lines.append("")

        # Memory model
        self.lines.append("OpMemoryModel Logical GLSL450")
        self.lines.append("")

        # Entry point (will be updated with actual function ID)
        self.lines.append('OpEntryPoint GLCompute %10 "main"')
        self.lines.append("OpExecutionMode %10 LocalSize 1 1 1")
        self.lines.append("")

    def emit_types(self) -> None:
        """Emit type declarations."""
        self.lines.append("; Type declarations")

        # Basic types
        uint_id = self.get_type_id("uint")
        self.lines.append(f"%{uint_id} = OpTypeInt 32 0")

        bool_id = self.get_type_id("bool")
        self.lines.append(f"%{bool_id} = OpTypeBool")

        void_id = self.get_type_id("void")
        self.lines.append(f"%{void_id} = OpTypeVoid")

        # Pointer types
        ptr_uint_id = self.get_type_id("ptr_uint")
        self.lines.append(f"%{ptr_uint_id} = OpTypePointer Function %{uint_id}")

        # Storage buffer pointer
        storage_ptr_uint_id = self.get_type_id("storage_ptr_uint")
        self.lines.append(
            f"%{storage_ptr_uint_id} = OpTypePointer StorageBuffer %{uint_id}"
        )

        # Function type: void(void)
        func_id = self.get_type_id("func_void")
        self.lines.append(f"%{func_id} = OpTypeFunction %{void_id}")

        self.lines.append("")

    def emit_constants(self) -> None:
        """Emit constant declarations."""
        self.lines.append("; Constants")

        uint_id = self.get_type_id("uint")

        # Common constants
        for val in [0, 1, 0xFFFFFFFF]:
            const_id = self.fresh_id()
            self.const_ids[val] = const_id
            self.lines.append(f"%{const_id} = OpConstant %{uint_id} {val}")

        self.lines.append("")

    def decompose_ternary(
        self, dst_id: int, a_id: int, b_id: int, c_id: int, imm8: int
    ) -> None:
        """
        Decompose VPTERNLOG/lop3 ternary operation to basic bitwise ops.

        Ternary logic table defined by imm8 (8-bit truth table).
        Formula: result[i] = LUT[{a[i], b[i], c[i]}]

        Common patterns:
        - imm8=0xCA (202): (a & b) | (c & ~b)  -- "blend"
        - imm8=0xD8 (216): (a & b) | (a & c) | (b & c)  -- "majority"
        - imm8=0x96 (150): a ^ b ^ c  -- "XOR3"
        - imm8=0x6A (106): a ^ (b & c)  -- "XOR_AND"
        """
        uint_id = self.get_type_id("uint")

        # For now, implement common cases
        if imm8 == 150:  # a ^ b ^ c
            temp1_id = self.fresh_id()
            self.lines.append(f"%{temp1_id} = OpBitwiseXor %{uint_id} %{a_id} %{b_id}")
            self.lines.append(
                f"%{dst_id} = OpBitwiseXor %{uint_id} %{temp1_id} %{c_id}"
            )
        elif imm8 == 106:  # a ^ (b & c)
            temp1_id = self.fresh_id()
            self.lines.append(f"%{temp1_id} = OpBitwiseAnd %{uint_id} %{b_id} %{c_id}")
            self.lines.append(
                f"%{dst_id} = OpBitwiseXor %{uint_id} %{a_id} %{temp1_id}"
            )
        elif imm8 == 40:  # (a & b) ^ c
            temp1_id = self.fresh_id()
            self.lines.append(f"%{temp1_id} = OpBitwiseAnd %{uint_id} %{a_id} %{b_id}")
            self.lines.append(
                f"%{dst_id} = OpBitwiseXor %{uint_id} %{temp1_id} %{c_id}"
            )
        elif imm8 == 128:  # a & b & c
            temp1_id = self.fresh_id()
            self.lines.append(f"%{temp1_id} = OpBitwiseAnd %{uint_id} %{a_id} %{b_id}")
            self.lines.append(
                f"%{dst_id} = OpBitwiseAnd %{uint_id} %{temp1_id} %{c_id}"
            )
        else:
            # Generic decomposition using truth table
            # result = 0
            # for each bit position in imm8:
            #   if bit i is set, OR in the corresponding term
            result_id = self.const_ids[0]

            for i in range(8):
                if imm8 & (1 << i):
                    # Decode which inputs are set in this row
                    # i = cba in binary: bit 0=a, bit 1=b, bit 2=c
                    mask_a = (i & 1) != 0
                    mask_b = (i & 2) != 0
                    mask_c = (i & 4) != 0

                    # Build term: (a if mask_a else ~a) & (b if mask_b else ~b) & (c if mask_c else ~c)
                    term_id = self.const_ids[0xFFFFFFFF]  # Start with all 1s

                    if mask_a:
                        temp_id = self.fresh_id()
                        self.lines.append(
                            f"%{temp_id} = OpBitwiseAnd %{uint_id} %{term_id} %{a_id}"
                        )
                        term_id = temp_id
                    else:
                        not_a_id = self.fresh_id()
                        self.lines.append(f"%{not_a_id} = OpNot %{uint_id} %{a_id}")
                        temp_id = self.fresh_id()
                        self.lines.append(
                            f"%{temp_id} = OpBitwiseAnd %{uint_id} %{term_id} %{not_a_id}"
                        )
                        term_id = temp_id

                    if mask_b:
                        temp_id = self.fresh_id()
                        self.lines.append(
                            f"%{temp_id} = OpBitwiseAnd %{uint_id} %{term_id} %{b_id}"
                        )
                        term_id = temp_id
                    else:
                        not_b_id = self.fresh_id()
                        self.lines.append(f"%{not_b_id} = OpNot %{uint_id} %{b_id}")
                        temp_id = self.fresh_id()
                        self.lines.append(
                            f"%{temp_id} = OpBitwiseAnd %{uint_id} %{term_id} %{not_b_id}"
                        )
                        term_id = temp_id

                    if mask_c:
                        temp_id = self.fresh_id()
                        self.lines.append(
                            f"%{temp_id} = OpBitwiseAnd %{uint_id} %{term_id} %{c_id}"
                        )
                        term_id = temp_id
                    else:
                        not_c_id = self.fresh_id()
                        self.lines.append(f"%{not_c_id} = OpNot %{uint_id} %{c_id}")
                        temp_id = self.fresh_id()
                        self.lines.append(
                            f"%{temp_id} = OpBitwiseAnd %{uint_id} %{term_id} %{not_c_id}"
                        )
                        term_id = temp_id

                    # OR into result
                    new_result_id = self.fresh_id()
                    self.lines.append(
                        f"%{new_result_id} = OpBitwiseOr %{uint_id} %{result_id} %{term_id}"
                    )
                    result_id = new_result_id

            # Copy final result to destination
            self.lines.append(f"%{dst_id} = OpCopyObject %{uint_id} %{result_id}")

    def emit_function(self, mir: MIRFunction) -> None:
        """Emit MIR function as SPIR-V."""
        self.lines.append("; Function body")

        void_id = self.get_type_id("void")
        func_id = self.get_type_id("func_void")
        uint_id = self.get_type_id("uint")

        # Function declaration
        func_body_id = self.fresh_id()
        self.lines.append(f"%{func_body_id} = OpFunction %{void_id} None %{func_id}")

        # Entry label
        label_id = self.fresh_id()
        self.lines.append(f"%{label_id} = OpLabel")
        self.lines.append("")

        # Allocate space for input registers
        self.lines.append("; Input registers (function parameters)")
        for i, reg in enumerate(mir.input_regs):
            reg_id = self.get_reg_id(reg)
            # For now, initialize to zero (would be loaded from buffers in compute shader)
            zero_id = self.const_ids[0]
            self.lines.append(
                f"%{reg_id} = OpCopyObject %{uint_id} %{zero_id}  ; input {i}"
            )
        self.lines.append("")

        # Emit instructions
        for inst in mir.instructions:
            if inst.dst:
                dst_id = self.get_reg_id(inst.dst)

            if isinstance(inst, Binary):
                a_id = self.get_reg_id(inst.a)
                b_id = self.get_reg_id(inst.b)

                if inst.op == "and":
                    self.lines.append(
                        f"%{dst_id} = OpBitwiseAnd %{uint_id} %{a_id} %{b_id}"
                    )
                elif inst.op == "or":
                    self.lines.append(
                        f"%{dst_id} = OpBitwiseOr %{uint_id} %{a_id} %{b_id}"
                    )
                elif inst.op == "xor":
                    self.lines.append(
                        f"%{dst_id} = OpBitwiseXor %{uint_id} %{a_id} %{b_id}"
                    )

            elif isinstance(inst, Unary):
                a_id = self.get_reg_id(inst.a)

                if inst.op == "not":
                    self.lines.append(f"%{dst_id} = OpNot %{uint_id} %{a_id}")

            elif isinstance(inst, Ternary):
                a_id = self.get_reg_id(inst.a)
                b_id = self.get_reg_id(inst.b)
                c_id = self.get_reg_id(inst.c)
                self.decompose_ternary(dst_id, a_id, b_id, c_id, inst.imm8)

            elif isinstance(inst, Copy):
                src_id = self.get_reg_id(inst.src)
                self.lines.append(f"%{dst_id} = OpCopyObject %{uint_id} %{src_id}")

            elif isinstance(inst, Const):
                const_val = inst.value
                if const_val not in self.const_ids:
                    const_id = self.fresh_id()
                    self.const_ids[const_val] = const_id
                    self.lines.insert(
                        -5, f"%{const_id} = OpConstant %{uint_id} {const_val}"
                    )
                const_id = self.const_ids[const_val]
                self.lines.append(f"%{dst_id} = OpCopyObject %{uint_id} %{const_id}")

        self.lines.append("")
        self.lines.append("OpReturn")
        self.lines.append("OpFunctionEnd")

    def finalize(self) -> str:
        """Finalize and return SPIR-V assembly."""
        # Update bound
        max_id = self.id_counter
        for i, line in enumerate(self.lines):
            if "Bound:" in line:
                self.lines[i] = f"; Bound: {max_id}"
                break

        return "\n".join(self.lines)


def emit_spirv(
    mir: MIRFunction, allocation: RegAllocation, function_name: str = "main"
) -> str:
    """
    Emit SPIR-V assembly from MIR.

    Args:
        mir: Machine IR function
        allocation: Register allocation info
        function_name: Name for the function (unused in SPIR-V, kept for API compat)

    Returns:
        SPIR-V assembly text
    """
    emitter = SPIRVEmitter()

    # Emit header
    num_regs = len(allocation.reg_assignment) + len(allocation.spills)
    emitter.emit_header(num_regs)

    # Emit types and constants
    emitter.emit_types()
    emitter.emit_constants()

    # Emit function
    emitter.emit_function(mir)

    return emitter.finalize()
