"""
Register allocation for scheduled circuits using linear scan.

This module provides:
- RegAllocation for complete allocation results
- LinearScanAllocator implementing the classic linear scan algorithm
"""

from dataclasses import dataclass, field

from stc.sched.schedule import Schedule
from stc.sched.liveness import LiveRange


@dataclass
class RegAllocation:
    """
    Complete register allocation result.

    reg_assignment maps node indices to physical registers.
    spills lists nodes that were spilled.
    loads and stores describe memory traffic as (node, reg, cycle) tuples.
    """

    reg_assignment: dict[int, int] = field(default_factory=dict)
    spills: list[int] = field(default_factory=list)
    loads: list[tuple[int, int, int]] = field(default_factory=list)
    stores: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def num_spills(self) -> int:
        return len(self.spills)


class LinearScanAllocator:
    """
    Linear scan register allocation.

    A simple, fast algorithm that processes live ranges in order of start
    cycle. When no register is available, it spills the interval with the
    furthest end point (longest remaining lifetime).
    """

    def __init__(self, num_registers: int):
        """
        Initialize the allocator.

        Args:
            num_registers: Number of physical registers available.
        """
        self._num_registers = num_registers

    @property
    def num_registers(self) -> int:
        return self._num_registers

    def allocate(
        self,
        live_ranges: dict[int, LiveRange],
        schedule: Schedule,
    ) -> RegAllocation:
        """
        Perform linear scan register allocation.

        Args:
            live_ranges: Dict mapping node index to LiveRange.
            schedule: The schedule assigning gates to cycles.

        Returns:
            RegAllocation with register assignments and spill operations.
        """
        if not live_ranges:
            return RegAllocation()

        sorted_ranges = sorted(live_ranges.values(), key=lambda r: (r.start, r.end))

        reg_assignment: dict[int, int] = {}
        spills: list[int] = []
        loads: list[tuple[int, int, int]] = []
        stores: list[tuple[int, int, int]] = []

        active: list[LiveRange] = []
        free_regs: set[int] = set(range(self._num_registers))
        node_to_slot: dict[int, int] = {}

        for current in sorted_ranges:
            self._expire_old_intervals(
                current.start,
                active,
                free_regs,
                reg_assignment,
            )

            if free_regs:
                reg = min(free_regs)
                free_regs.remove(reg)
                reg_assignment[current.node] = reg
                active.append(current)
                active.sort(key=lambda r: r.end)
            else:
                spill_target, reg = self._spill_at_interval(
                    current,
                    active,
                    reg_assignment,
                    spills,
                    loads,
                    stores,
                    node_to_slot,
                )

                if spill_target is not None:
                    reg_assignment[current.node] = reg
                    active.append(current)
                    active.sort(key=lambda r: r.end)
                else:
                    spills.append(current.node)
                    stores.append((current.node, -1, current.start))

        return RegAllocation(
            reg_assignment=reg_assignment,
            spills=spills,
            loads=loads,
            stores=stores,
        )

    def _expire_old_intervals(
        self,
        current_start: int,
        active: list[LiveRange],
        free_regs: set[int],
        node_to_reg: dict[int, int],
    ) -> None:
        """Remove intervals that have ended before current_start."""
        expired = []
        for r in active:
            if r.end < current_start:
                expired.append(r)

        for r in expired:
            active.remove(r)
            reg = node_to_reg.get(r.node)
            if reg is not None:
                free_regs.add(reg)

    def _spill_at_interval(
        self,
        current: LiveRange,
        active: list[LiveRange],
        reg_assignment: dict[int, int],
        spills: list[int],
        loads: list[tuple[int, int, int]],
        stores: list[tuple[int, int, int]],
        node_to_slot: dict[int, int],
    ) -> tuple[LiveRange | None, int]:
        """
        Handle the case where no register is available.

        Either spill the longest-remaining active interval (if it ends after
        current), or spill current itself.

        Returns (spilled_range, assigned_reg).
        If spilled_range is None, current was spilled and has no register.
        """
        if not active:
            return (None, -1)

        longest = max(active, key=lambda r: r.end)

        if longest.end > current.end:
            reg = reg_assignment[longest.node]

            stores.append((longest.node, reg, current.start))

            active.remove(longest)
            del reg_assignment[longest.node]
            spills.append(longest.node)

            loads.append((longest.node, reg, longest.end))

            return (longest, reg)

        return (None, -1)


def allocate_registers(
    live_ranges: dict[int, LiveRange],
    schedule: Schedule,
    num_registers: int,
) -> RegAllocation:
    """
    Convenience function for linear scan register allocation.

    Args:
        live_ranges: Dict mapping node index to LiveRange.
        schedule: The schedule assigning gates to cycles.
        num_registers: Number of physical registers available.

    Returns:
        RegAllocation with register assignments and spill operations.
    """
    allocator = LinearScanAllocator(num_registers)
    return allocator.allocate(live_ranges, schedule)
