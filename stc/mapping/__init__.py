"""Technology mapping modules."""

from stc.mapping.ternary import (
    compute_imm8,
    extract_3input_cone,
    TernaryMappingPass,
)

__all__ = ["compute_imm8", "extract_3input_cone", "TernaryMappingPass"]
