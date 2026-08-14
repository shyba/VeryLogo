from __future__ import annotations

from dataclasses import dataclass

from stc.yosys_json import YosysDesign, is_gemm_cell


@dataclass(frozen=True)
class SubsetError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


SUPPORTED_CELL_TYPES = {
    "$not",
    "$logic_not",
    "$and",
    "$or",
    "$xor",
    "$mux",
    "$pmux",
    "$add",
    "$sub",
    "$mul",
    "$shl",
    "$shr",
    "$sshr",
    "$eq",
    "$ne",
    "$lt",
    "$le",
    "$gt",
    "$ge",
    "$mem_v2",
    "$memrd",
    "$meminit",
    "$meminit_v2",
    "$dff",
    "$dffe",
    "$sdff",
    "$sdffe",
    "$reduce_or",
    "$reduce_bool",
    "$scopeinfo",
    "$_NOT_",
    "$_AND_",
    "$_OR_",
    "$_XOR_",
    "$_XNOR_",
    "$_NAND_",
    "$_NOR_",
    "$_MUX_",
    "$_ANDNOT_",
    "$_ORNOT_",
    "$_SDFF_PP0_",
    "$_SDFFE_PP0P_",
    "$_SDFFE_PP0N_",
    "$lut",
}


def _parse_param_int(raw: str) -> int:
    s = raw.strip()
    if not s:
        raise ValueError("empty param")
    if set(s) <= {"0", "1"}:
        return int(s, 2)
    return int(s, 10)


def _param_bool(params: dict[str, str], key: str) -> bool:
    raw = params.get(key)
    if raw is None:
        return False
    return _parse_param_int(raw) != 0


def _param_u32(params: dict[str, str], key: str) -> int | None:
    raw = params.get(key)
    if raw is None:
        return None
    return _parse_param_int(raw)


def check_subset(design: YosysDesign) -> None:
    module = design.modules[design.top]
    for cell in module.cells.values():
        if cell.type in {"$adff", "$adffe", "$dffsr", "$dffr", "$dffsre"}:
            raise SubsetError(f"async reset/set not supported: {cell.type}")
        if (
            not cell.type.startswith("$")
            and cell.type in design.modules
            and cell.type != design.top
        ):
            if not is_gemm_cell(design, cell):
                raise SubsetError(f"submodule instantiation not supported: {cell.type}")
        if is_gemm_cell(design, cell):
            # GEMM is a shaped primitive.  Its full contract is checked by
            # extraction/type inference; keeping the subset gate here makes
            # hierarchy lifting explicit while rejecting unknown submodules.
            continue
        if cell.type not in SUPPORTED_CELL_TYPES:
            raise SubsetError(f"unsupported cell type: {cell.type}")

        params = cell.parameters
        if cell.type in {
            "$add",
            "$sub",
            "$mul",
            "$shl",
            "$shr",
            "$lt",
            "$le",
            "$gt",
            "$ge",
        }:
            if _param_bool(params, "A_SIGNED") or _param_bool(params, "B_SIGNED"):
                raise SubsetError(f"unsupported signed cell params for {cell.type}")
        if cell.type == "$sshr":
            if not _param_bool(params, "A_SIGNED") or _param_bool(params, "B_SIGNED"):
                raise SubsetError("sshr requires A_SIGNED=1 and B_SIGNED=0")

        if cell.type in {"$lt", "$le", "$gt", "$ge"}:
            y_width = _param_u32(params, "Y_WIDTH")
            if y_width is not None and y_width != 1:
                raise SubsetError(f"{cell.type} requires Y_WIDTH=1")
        if cell.type == "$dff":
            if "D" not in cell.connections or "Q" not in cell.connections:
                raise SubsetError("dff requires D and Q ports")
        if cell.type == "$dffe":
            required = {"D", "Q", "EN", "CLK"}
            if not required.issubset(set(cell.connections.keys())):
                raise SubsetError("dffe requires D, Q, EN, and CLK ports")
            if _param_u32(params, "CLK_POLARITY") not in {None, 1}:
                raise SubsetError("dffe requires CLK_POLARITY=1")
            if _param_u32(params, "EN_POLARITY") not in {None, 1}:
                raise SubsetError("dffe requires EN_POLARITY=1")
        if cell.type == "$sdff":
            required = {"D", "Q", "SRST", "CLK"}
            if not required.issubset(set(cell.connections.keys())):
                raise SubsetError("sdff requires D, Q, SRST, and CLK ports")
        if cell.type == "$sdffe":
            required = {"D", "Q", "SRST", "CLK", "EN"}
            if not required.issubset(set(cell.connections.keys())):
                raise SubsetError("sdffe requires D, Q, SRST, CLK, and EN ports")

        if cell.type == "$mem_v2":
            width = _param_u32(params, "WIDTH")
            abits = _param_u32(params, "ABITS")
            size = _param_u32(params, "SIZE")
            rd_ports = _param_u32(params, "RD_PORTS")
            wr_ports = _param_u32(params, "WR_PORTS")
            if width not in {None, 8}:
                raise SubsetError("mem_v2 supports only WIDTH=8")
            if abits not in {None, 8}:
                raise SubsetError("mem_v2 supports only ABITS=8")
            if size not in {None, 256}:
                raise SubsetError("mem_v2 supports only SIZE=256")
            if rd_ports is None or rd_ports < 1:
                raise SubsetError("mem_v2 requires RD_PORTS >= 1")
            if wr_ports not in {None, 0}:
                raise SubsetError("mem_v2 does not support writes (WR_PORTS=0)")
            if _param_u32(params, "RD_CLK_ENABLE") not in {None, 0}:
                raise SubsetError("mem_v2 does not support read clocks")

        if cell.type == "$meminit_v2":
            width = _param_u32(params, "WIDTH")
            words = _param_u32(params, "WORDS")
            if width not in {None, 8}:
                raise SubsetError("meminit_v2 supports only WIDTH=8")
            if words not in {None, 256}:
                raise SubsetError("meminit_v2 supports only WORDS=256")

        if cell.type == "$meminit":
            width = _param_u32(params, "WIDTH")
            words = _param_u32(params, "WORDS")
            if width not in {None, 8}:
                raise SubsetError("meminit supports only WIDTH=8")
            if words not in {None, 256}:
                raise SubsetError("meminit supports only WORDS=256")

        if cell.type == "$memrd":
            width = _param_u32(params, "WIDTH")
            abits = _param_u32(params, "ABITS")
            if width not in {None, 8}:
                raise SubsetError("memrd supports only WIDTH=8")
            if abits not in {None, 8}:
                raise SubsetError("memrd supports only ABITS=8")
            if _param_u32(params, "CLK_ENABLE") not in {None, 0}:
                raise SubsetError("memrd does not support clocks")
            if _param_u32(params, "CLK_POLARITY") not in {None, 0}:
                raise SubsetError("memrd does not support clocks")
