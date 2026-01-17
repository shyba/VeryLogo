This repository contains two Markdown documents. No fenced code blocks are used.

# Document 1 — Project Overview

# Space-Time Compiler: HDL → CPU / GPU / TPU

## 1. Overview

This project proposes a new kind of compiler that starts from HDL descriptions (initially a restricted subset of Verilog) and transforms them into explicit space-time machines that can execute on multiple targets: microcontrollers, CPUs, GPUs, and TPUs.

Unlike traditional software compilers or functional frameworks (e.g., JAX), time, state, and physical resources are first-class elements of the program semantics. The code describes not only what to compute, but when to compute it and with which resources.

Conceptually, the system sits between:

* hardware synthesis (HLS)
* firmware compilers
* mapping temporal machines onto modern parallel architectures

## 2. Motivation

Real-world problems in systems, control, DSP, robotics, and reactive pipelines tend to be:

* strongly temporal
* state-oriented
* latency- and predictability-sensitive
* hard to express in purely functional models

HDL already captures these aspects in the hardware domain. This project explores the hypothesis that HDL can act as a universal intermediate language, able to generate:

* hardware
* firmware
* GPU kernels
* specialized pipelines
  with formal guarantees inherited from logic synthesis.

## 3. Core Idea

Every program is modeled as a discrete state transition:

S' = f(S, I)
O  = g(S, I)

Where:

* S is explicit state (registers)
* I are inputs (GPIO, memory, streams)
* O are observable outputs
* tick (clock) is a real semantic unit

This representation enables:

* explicit time scheduling (tracks / lanes)
* latency modeling
* solver-driven reduction and equivalence
* predictable code generation

## 4. Intermediate Representation (IR)

The project IR is temporal and sequential, not only functional:

* operations have latency
* data and time dependencies are explicit
* state persists across ticks
* resources are modelable

The IR can be viewed as:

* an enriched FSM
* an algorithmic form of a sequential circuit
* a state machine with a datapath

## 5. Space-Time Track

Execution is visualized as a 2D track:

* X-axis: time (ticks / cycles)
* Y-axis: resources (lanes, ALUs, warps, stages)

Each operation occupies:

* a width (resource usage)
* a time extent (latency)

This track is:

* executable (backend)
* inspectable (debug)
* optimizable (solver + scheduler)

## 6. Solvers

Because the IR is equivalent to a sequential circuit, the project integrates classic EDA techniques:

* boolean simplification
* state reduction
* equivalence checking
* constraint-guided synthesis

SMT/SAT solvers are used to:

* reduce logic
* remove dead state
* prove correctness of optimizations
* search for cheaper implementations

## 7. Planned Backends

### Microcontrollers (e.g., AVR)

* 1 lane
* tick execution by polling
* predictable C or assembly

### CPU

* lanes as functional units
* C, LLVM, or assembly emission
* focus on deterministic latency

### GPU (PTX)

* threads and warps as lanes
* explicit SIMT pipelines
* kernels as ticks or stages

### TPU

* mapping to systolic arrays
* explicit control outside matmul
* deterministic inference

## 8. Final Goals

1. Demonstrate HDL as a universal space-time IR.
2. Generate efficient, predictable code for multiple targets.
3. Integrate solvers as a central compilation stage.
4. Unify concepts from hardware, firmware, and parallel compute.
5. Produce a tool that is explainable, verifiable, and extensible.

## 9. High-Level Roadmap

### Phase 1 — MVP

* Verilog subset
* tick-based IR
* AVR backend
* track visualization (minimal)

### Phase 2 — Solver

* combinational simplification
* dead-state removal
* equivalence proofs

### Phase 3 — GPU

* experimental PTX backend
* SIMT mapping
* warp-level pipelines

### Phase 4 — Generalization

* CPU multi-lane
* experimental TPU backend
* multi-objective optimization

## 10. Status

This project is in an advanced technical conception phase, with a defined architecture and initial focus on experimental validation of the core hypothesis.

---

# Document 2 — Technical Validation Plan (MVP + Metrics)

## 1. Validation Objective

Validate the central hypothesis:

“HDL descriptions, when compiled as explicit space-time machines, enable optimizations, checks, and mappings that are not viable in traditional compilers or purely functional frameworks.”

The validation targets correctness, explainability, and structural advantage, not absolute performance.

## 2. MVP Validation Scope

### Included

* Verilog subset (single module, single clock)
* `always_ff` with reset
* basic arithmetic and logic operations
* input/output GPIO
* persistent state
* ATtiny85 backend
* basic combinational and sequential solver stages

### Excluded

* multi-lane parallelism
* GPU/TPU
* complex memories
* aggressive optimization

## 3. Main Test Cases

### Case 1 — Combinational logic

* direct input-to-output logic
* verify reduction to simpler expressions
* measure elimination of unnecessary state

### Case 2 — Simple FSM

* button-controlled toggle
* compare original vs minimized FSM
* measure register/state reduction

### Case 3 — Temporal pipeline

* counter + PWM
* verify explicit latency
* confirm equivalence after reduction

### Case 4 — Debounce

* intentionally redundant state
* verify the solver removes unreachable/dead states

## 4. MVP Implementation Pipeline

### Step 1 — Frontend

* parse the Verilog subset via Yosys
* extract state, IO, and `always_ff` structure
* emit clear errors when outside the subset

### Step 2 — Tick IR

* normalize into `S' = f(S, I)`
* explicit state representation
* separate logic from effects

### Step 3 — Solver

* bit-blast expressions
* boolean simplification
* reachability analysis
* dead-state removal

### Step 4 — Verification

* compare original vs reduced IR
* bounded equivalence checking
* Python golden model

### Step 5 — AVR Backend

* generate predictable C
* shadow PORT writes
* tick execution in a loop

## 5. Evaluation Metrics

### Structural metrics

* register count before/after
* operation count per tick
* estimated combinational depth

### Generated-code metrics

* C LOC
* approximate AVR instruction count
* RAM/FLASH usage

### Semantic metrics

* automatically proven functional equivalence
* tick-level determinism
* no undefined behavior

### Qualitative metrics

* clarity of the time track
* ability to explain behavior
* ease of spotting redundancy

## 6. Success Criteria

The MVP is considered validated if:

1. A non-trivial HDL design generates working firmware.
2. The solver reduces logic or state measurably.
3. Equivalence is automatically proven (bounded).
4. Temporal behavior is predictable.
5. The track representation matches generated code.

## 7. Technical Risks

* solver state explosion
* balancing generality vs simplicity
* premature over-engineering
* poorly defined HDL subset

Mitigations:

* bounded proofs
* block-local reductions
* strict MVP scope

## 8. Expected Outcomes

* a concrete proof that HDL can act as a universal IR
* a measurable formal reduction in “programs”
* a solid base for GPU/TPU work
* technical material for academic/industrial discussion

## 9. Next Steps After Validation

* introduce multiple lanes
* experimental PTX backend
* cost-guided synthesis
* explore DSP/control applications

## 10. Conclusion

This plan validates not only a compiler, but a paradigm shift: programs as explicit machines that are optimizable and verifiable with the same tools used in hardware for decades.
