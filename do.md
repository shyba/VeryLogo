> Fixed context (do not reopen decisions):
This project uses HDL (a Verilog subset) as the source of truth to generate a space-time IR (Tick-IR) that represents explicit sequential machines.
`info.md` defines scope and negative decisions
`research.md` contains tooling research (Yosys, solvers, etc.).
This document (`design.md`) must define only the implementable MVP.

Do not discuss alternatives, do not ask preferences, do not reopen decisions.
Assume the MVP uses Yosys as the frontend, Python as the implementation language, and ATtiny85 as the initial backend.


---

Task

Create a `design.md` document that describes, concretely and executably, the technical MVP of the space-time compiler.

The document must contain exactly the sections below, in this order, using objective technical language.


---

Required structure for `design.md`

1. MVP Objective

What the MVP proves (technical hypothesis).

What is not part of the MVP.


2. Compilation Pipeline (Overview)

Describe the fixed flow:

input.v
 → normalized.(rtlil/json)
 → tick_ir.bin
 → reduced_tick_ir.bin
 → avr.c

Briefly explain the role of each stage.


---

3. Verilog Frontend (Yosys)

Yosys role: parse, elaboration, and normalization, not aggressive optimization.

Allowed pass types.

Chosen output format (RTLIL or JSON).

Accepted Verilog subset restrictions.


---

4. Tick-IR (Source of Truth)

Formally define Tick-IR, including:

Semantic model:

S' = f(S, I)
O  = g(S, I)

Supported types (bitvec, bool).

What counts as state.

What counts as combinational logic.

How “delay / pipeline / register” is represented.

Tick-IR guarantees (determinism, simultaneous updates).


---

5. Scheduling and Time Track

How Tick-IR operations become a track (1-lane in the MVP).

What latency means in the MVP.

What is not implemented yet (e.g., multi-lane, throughput > 1).


---

6. Reduction and Solvers

Where the solver fits in the pipeline.

Allowed reduction types in the MVP:

combinational

dead state

What must be preserved (functional equivalence).

Validation method (bounded).


---

7. ATtiny85 Backend

Execution model (tick = loop or timer ISR).

Tick-IR state → C variables mapping.

I/O mapping (GPIO).

Code generation rules (branchless when possible).


---

8. MVP Validation

Functional simulation (Python + Verilator).

Golden model.

Bounded equivalence (optional, via Yosys/SymbiYosys).

Objective “MVP validated” criteria.


---

9. Artifacts and Debug

Generated, versionable files.

What must be manually inspectable.

Minimal logs and metrics.


---

10. Out of Scope (Explicitly)

Short, clear list of what is not part of the MVP:

GPU / PTX

Multi-lane

Performance tuning

Custom Verilog frontend

UI / advanced visualization


---

Final rules (mandatory)

Do not suggest alternatives.

Do not use vague language (“maybe”, “could”).

Write as real-project technical documentation.

Output only the contents of `design.md`.


