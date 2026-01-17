Prototype: Optimized Verilog-to-Pipeline Compilation

Concept Overview

The idea is to treat HDL code (e.g., Verilog) as an algorithmic description and schedule it over time under limited hardware resources. Instead of compiling to conventional software, the system maps operations to time-ordered “boxes” (arithmetic/logic operations) connected by data dependencies on a time track. Track width corresponds to the number of functional units available in parallel, while track length corresponds to time (clock cycles). Each operation occupies a time interval proportional to its latency and must respect data dependencies (outputs feeding inputs of later operations). This matches the core of High-Level Synthesis (HLS): determine in which cycle each operation executes under resource limits. The result is a scheduled pipeline that exploits parallelism up to the available resources.

Example (2 ALUs): in cycle 0, two independent adds execute in parallel. Their results feed a multiply that starts in cycle 1 and occupies cycles 1–2 if its latency is 2. In a minimal prototype, each resource starts a new operation only after the previous one completes (initiation interval optimization is out of scope).

Parsing and HDL Representation Tooling

For a prototype, Python is a practical choice because it has HDL analysis libraries. One option is PyVerilog, which provides a Verilog parser and an AST, along with dataflow analysis utilities. This enables extracting a dependency graph (DFG) of operations and implementing a custom scheduler. Yosys is an alternative frontend that can produce a normalized internal representation or netlists that can also be consumed by custom tooling.

With an operation graph in hand, classic compiler optimizations are possible, but the core focus is scheduling. Scheduling under resource constraints is a well-studied problem in HLS and can reuse known algorithms (e.g., list scheduling) and bounded constraints for small test cases.

Pipeline Generation and Validation

Given a schedule (operation → start cycle + lane), a next step is to generate a hardware model. A simple approach is to emit Verilog with explicit pipeline registers and/or an FSM where each state corresponds to a cycle, committing intermediate results into registers. Functional validation can be automated using simulators such as Icarus Verilog or Verilator. Python test harnesses can drive tick-by-tick stimuli and compare outputs against a reference.

Formal validation can be done with equivalence checking or bounded model checking using Yosys + SMT (e.g., SymbiYosys), but it is not required for a minimal prototype.

Recommended Language and Environment

Python enables fast iteration for parsing/analysis, scheduling, code generation, and test automation. It integrates well with HDL tooling and supports quick proof-of-concept development.
