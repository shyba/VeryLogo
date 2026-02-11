use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "avx512")]
    target: String,
    #[arg(long, default_value = "list")]
    scheduler: String,
    #[arg(long, default_value = "circuit")]
    function_name: String,
    #[arg(long)]
    input_io_bits: Option<u32>,
    #[arg(long)]
    output_io_bits: Option<u32>,
    #[arg(long)]
    max_live_pressure: Option<u32>,
    #[arg(long, default_value = "json")]
    format: String,
    #[arg(long)]
    timing: bool,
}

#[derive(Clone, Debug)]
enum Gate {
    Const { value: u32, width: u32 },
    Unary { op: String, a: u32 },
    Binary { op: String, a: u32, b: u32 },
    Ternary { a: u32, b: u32, c: u32, imm8: u8 },
}

#[derive(Clone, Debug)]
struct Circuit {
    input_bits: u32,
    output_bits: u32,
    gates: Vec<Gate>,
    outputs: Vec<(u32, bool)>,
}

fn parse_gate(v: &Value) -> Result<Gate> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("gate not array"))?;
    if arr.len() == 5 {
        let op = arr[0]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("gate op"))?;
        if op != "ternary" {
            bail!("unexpected 5-elem gate op {op}");
        }
        let a = arr[1].as_u64().ok_or_else(|| anyhow::anyhow!("gate a"))? as u32;
        let b = arr[2].as_u64().ok_or_else(|| anyhow::anyhow!("gate b"))? as u32;
        let c = arr[3].as_u64().ok_or_else(|| anyhow::anyhow!("gate c"))? as u32;
        let imm8 = arr[4]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("gate imm8"))? as u8;
        Ok(Gate::Ternary { a, b, c, imm8 })
    } else if arr.len() == 3 {
        let op = arr[0]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("gate op"))?
            .to_string();
        let a = arr[1].as_u64().ok_or_else(|| anyhow::anyhow!("gate a"))? as u32;
        let b = arr[2].as_u64().ok_or_else(|| anyhow::anyhow!("gate b"))? as u32;
        if op == "const" {
            Ok(Gate::Const { value: a, width: b })
        } else if op == "not" || op == "shl" || op == "lshr" {
            Ok(Gate::Unary { op, a })
        } else {
            Ok(Gate::Binary { op, a, b })
        }
    } else {
        bail!("unexpected gate length {}", arr.len());
    }
}

fn parse_circuit_json(path: &PathBuf) -> Result<Circuit> {
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let v: Value = serde_json::from_slice(&data)?;
    let input_bits = v
        .get("input_bits")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing input_bits"))? as u32;
    let output_bits = v
        .get("output_bits")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing output_bits"))? as u32;
    let gates_v = v
        .get("gates")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing gates"))?;
    let mut gates = Vec::with_capacity(gates_v.len());
    for g in gates_v {
        gates.push(parse_gate(g)?);
    }
    let outputs_v = v
        .get("outputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing outputs"))?;
    let mut outputs = Vec::with_capacity(outputs_v.len());
    for o in outputs_v {
        let arr = o
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("output not array"))?;
        if arr.len() != 2 {
            bail!("output len != 2");
        }
        let idx = arr[0]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("output idx"))? as u32;
        let inv = arr[1]
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("output inv"))?;
        outputs.push((idx, inv));
    }
    Ok(Circuit {
        input_bits,
        output_bits,
        gates,
        outputs,
    })
}

fn read_u8(data: &[u8], off: &mut usize) -> Result<u8> {
    if *off >= data.len() {
        bail!("unexpected EOF");
    }
    let v = data[*off];
    *off += 1;
    Ok(v)
}

fn read_u32(data: &[u8], off: &mut usize) -> Result<u32> {
    if *off + 4 > data.len() {
        bail!("unexpected EOF");
    }
    let v = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}

fn parse_circuit_bin(path: &PathBuf) -> Result<Circuit> {
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut off = 0usize;
    if data.len() < 5 {
        bail!("circuit bin too small");
    }
    let magic = &data[0..4];
    if magic != b"CSB1" {
        bail!("bad circuit bin magic");
    }
    off += 4;
    let version = read_u8(&data, &mut off)?;
    if version != 1 {
        bail!("unsupported circuit bin version {}", version);
    }
    let input_bits = read_u32(&data, &mut off)?;
    let output_bits = read_u32(&data, &mut off)?;
    let gate_count = read_u32(&data, &mut off)?;
    let output_count = read_u32(&data, &mut off)?;

    let mut gates = Vec::with_capacity(gate_count as usize);
    for _ in 0..gate_count {
        let opcode = read_u8(&data, &mut off)?;
        match opcode {
            0 => {
                let a = read_u32(&data, &mut off)?;
                let b = read_u32(&data, &mut off)?;
                gates.push(Gate::Binary {
                    op: "xor".to_string(),
                    a,
                    b,
                });
            }
            1 => {
                let a = read_u32(&data, &mut off)?;
                let b = read_u32(&data, &mut off)?;
                gates.push(Gate::Binary {
                    op: "and".to_string(),
                    a,
                    b,
                });
            }
            2 => {
                let a = read_u32(&data, &mut off)?;
                let b = read_u32(&data, &mut off)?;
                gates.push(Gate::Binary {
                    op: "or".to_string(),
                    a,
                    b,
                });
            }
            3 => {
                let a = read_u32(&data, &mut off)?;
                gates.push(Gate::Unary {
                    op: "not".to_string(),
                    a,
                });
            }
            4 => {
                let value = read_u32(&data, &mut off)?;
                let width = read_u32(&data, &mut off)?;
                gates.push(Gate::Const { value, width });
            }
            5 => {
                let a = read_u32(&data, &mut off)?;
                let b = read_u32(&data, &mut off)?;
                let c = read_u32(&data, &mut off)?;
                let imm8 = read_u8(&data, &mut off)?;
                gates.push(Gate::Ternary { a, b, c, imm8 });
            }
            6 => {
                let a = read_u32(&data, &mut off)?;
                let b = read_u32(&data, &mut off)?;
                gates.push(Gate::Binary {
                    op: "andnot".to_string(),
                    a,
                    b,
                });
            }
            7 => {
                let a = read_u32(&data, &mut off)?;
                let b = read_u32(&data, &mut off)?;
                gates.push(Gate::Binary {
                    op: "ornot".to_string(),
                    a,
                    b,
                });
            }
            other => bail!("unknown gate opcode {}", other),
        }
    }

    let mut outputs = Vec::with_capacity(output_count as usize);
    for _ in 0..output_count {
        let idx = read_u32(&data, &mut off)?;
        let inv = read_u8(&data, &mut off)? != 0;
        outputs.push((idx, inv));
    }

    Ok(Circuit {
        input_bits,
        output_bits,
        gates,
        outputs,
    })
}

fn gate_operands(g: &Gate) -> Vec<u32> {
    match g {
        Gate::Ternary { a, b, c, .. } => vec![*a, *b, *c],
        Gate::Unary { a, .. } => vec![*a],
        Gate::Binary { a, b, .. } => vec![*a, *b],
        Gate::Const { .. } => vec![],
    }
}

#[derive(Clone, Debug)]
struct TargetModel {
    registers: u32,
    issue_width: u32,
    latencies: HashMap<String, u32>,
    throughput: HashMap<String, u32>,
}

#[derive(Clone, Debug)]
struct EmitOps {
    vec_ty: &'static str,
    setzero: &'static str,
    set1: &'static str,
    xor: &'static str,
    and: &'static str,
    or: &'static str,
    andnot: &'static str,
    ternary: &'static str,
}

impl TargetModel {
    fn latency(&self, op: &str) -> u32 {
        *self.latencies.get(op).unwrap_or(&1)
    }

    fn max_per_cycle(&self, op: &str) -> u32 {
        *self.throughput.get(op).unwrap_or(&self.issue_width)
    }
}

fn target_avx512() -> TargetModel {
    let mut lat = HashMap::new();
    let mut thr = HashMap::new();
    for op in ["xor", "and", "or", "not", "andn", "ternary"] {
        lat.insert(op.to_string(), 1);
        thr.insert(op.to_string(), 2);
    }
    TargetModel {
        registers: 32,
        issue_width: 4,
        latencies: lat,
        throughput: thr,
    }
}

fn target_avx2() -> TargetModel {
    let mut lat = HashMap::new();
    let mut thr = HashMap::new();
    for op in ["xor", "and", "or", "not", "andn", "ternary"] {
        lat.insert(op.to_string(), 1);
    }
    thr.insert("xor".to_string(), 3);
    thr.insert("and".to_string(), 2);
    thr.insert("or".to_string(), 2);
    thr.insert("not".to_string(), 2);
    thr.insert("andn".to_string(), 2);
    thr.insert("ternary".to_string(), 2);
    TargetModel {
        registers: 16,
        issue_width: 4,
        latencies: lat,
        throughput: thr,
    }
}

fn emit_ops(target: &str) -> Result<EmitOps> {
    match target {
        "avx512" => Ok(EmitOps {
            vec_ty: "__m512i",
            setzero: "_mm512_setzero_si512()",
            set1: "_mm512_set1_epi32(-1)",
            xor: "_mm512_xor_si512",
            and: "_mm512_and_si512",
            or: "_mm512_or_si512",
            andnot: "_mm512_andnot_si512",
            ternary: "_mm512_ternarylogic_epi32",
        }),
        "avx2" => Ok(EmitOps {
            vec_ty: "__m256i",
            setzero: "_mm256_setzero_si256()",
            set1: "_mm256_set1_epi32(-1)",
            xor: "_mm256_xor_si256",
            and: "_mm256_and_si256",
            or: "_mm256_or_si256",
            andnot: "_mm256_andnot_si256",
            ternary: "_mm256_ternarylogic_epi32",
        }),
        other => bail!("unsupported target {}", other),
    }
}

#[derive(Clone, Debug)]
struct DependencyInfo {
    predecessors: Vec<HashSet<usize>>,
    successors: Vec<HashSet<usize>>,
}

fn compute_dependencies(gates: &[Gate], input_bits: u32) -> DependencyInfo {
    let num_gates = gates.len();
    let mut predecessors = vec![HashSet::new(); num_gates];
    let mut successors = vec![HashSet::new(); num_gates];
    for (g_idx, gate) in gates.iter().enumerate() {
        let operands = gate_operands(gate);
        for operand in operands {
            if operand >= input_bits {
                let pred = (operand - input_bits) as usize;
                if pred < num_gates {
                    predecessors[g_idx].insert(pred);
                    successors[pred].insert(g_idx);
                }
            }
        }
    }
    DependencyInfo {
        predecessors,
        successors,
    }
}

fn topo_order(deps: &DependencyInfo) -> Vec<usize> {
    let num_gates = deps.predecessors.len();
    let mut in_degree: Vec<usize> = deps
        .predecessors
        .iter()
        .map(|p| p.len())
        .collect();
    let mut ready: Vec<usize> = (0..num_gates).filter(|g| in_degree[*g] == 0).collect();
    let mut order = Vec::with_capacity(num_gates);
    let mut idx = 0;
    while idx < ready.len() {
        let g = ready[idx];
        idx += 1;
        order.push(g);
        for succ in deps.successors[g].iter() {
            in_degree[*succ] -= 1;
            if in_degree[*succ] == 0 {
                ready.push(*succ);
            }
        }
    }
    order
}

fn compute_asap(gates: &[Gate], input_bits: u32, latencies: &HashMap<String, u32>) -> Vec<u32> {
    let deps = compute_dependencies(gates, input_bits);
    let order = topo_order(&deps);
    let mut asap = vec![0u32; gates.len()];
    for g_idx in order {
        let gate = &gates[g_idx];
        let operands = gate_operands(gate);
        let mut earliest = 0u32;
        for operand in operands {
            if operand >= input_bits {
                let pred = (operand - input_bits) as usize;
                let pred_op = gate_op_name(&gates[pred]);
                let pred_latency = *latencies.get(pred_op).unwrap_or(&1);
                let ready = asap[pred] + pred_latency;
                if ready > earliest {
                    earliest = ready;
                }
            }
        }
        asap[g_idx] = earliest;
    }
    asap
}

fn compute_alap(
    gates: &[Gate],
    input_bits: u32,
    outputs: &[(u32, bool)],
    latencies: &HashMap<String, u32>,
    max_cycles: u32,
) -> Vec<u32> {
    let deps = compute_dependencies(gates, input_bits);
    let order = topo_order(&deps);
    let mut alap = vec![0u32; gates.len()];
    let mut output_gates = HashSet::new();
    for (out_idx, _) in outputs {
        if *out_idx >= input_bits {
            output_gates.insert((*out_idx - input_bits) as usize);
        }
    }
    for g_idx in 0..gates.len() {
        if deps.successors[g_idx].is_empty() || output_gates.contains(&g_idx) {
            let op = gate_op_name(&gates[g_idx]);
            let latency = *latencies.get(op).unwrap_or(&1);
            alap[g_idx] = max_cycles.saturating_sub(latency);
        }
    }
    for &g_idx in order.iter().rev() {
        if alap[g_idx] != 0 || output_gates.contains(&g_idx) {
            continue;
        }
        let op = gate_op_name(&gates[g_idx]);
        let latency = *latencies.get(op).unwrap_or(&1);
        let mut latest = max_cycles;
        for succ in deps.successors[g_idx].iter() {
            let succ_alap = alap[*succ];
            if succ_alap >= latency {
                latest = latest.min(succ_alap - latency);
            }
        }
        alap[g_idx] = latest;
    }
    alap
}

fn compute_slack(asap: &[u32], alap: &[u32]) -> Vec<i32> {
    asap.iter()
        .zip(alap.iter())
        .map(|(a, b)| *b as i32 - *a as i32)
        .collect()
}

fn gate_op_name(gate: &Gate) -> &str {
    match gate {
        Gate::Const { .. } => "const",
        Gate::Unary { op, .. } => op.as_str(),
        Gate::Binary { op, .. } => op.as_str(),
        Gate::Ternary { .. } => "ternary",
    }
}

#[derive(Clone, Debug)]
struct Schedule {
    gate_cycle: Vec<u32>,
}

impl Schedule {
    fn total_cycles(&self) -> u32 {
        self.gate_cycle.iter().max().copied().unwrap_or(0) + 1
    }
}

fn list_schedule(
    gates: &[Gate],
    input_bits: u32,
    outputs: &[(u32, bool)],
    target: &TargetModel,
    max_live_pressure: Option<u32>,
    timing: bool,
) -> Schedule {
    if gates.is_empty() {
        return Schedule { gate_cycle: vec![] };
    }
    let t_start = Instant::now();
    let mut last_report = Instant::now();
    let latencies = &target.latencies;
    let asap = compute_asap(gates, input_bits, latencies);
    let max_depth = asap.iter().max().copied().unwrap_or(0) + 1;
    let alap = compute_alap(gates, input_bits, outputs, latencies, max_depth * 2);
    let slack = compute_slack(&asap, &alap);
    let deps = compute_dependencies(gates, input_bits);

    let mut ready_at: Vec<Option<u32>> = vec![None; gates.len()];
    for g in 0..gates.len() {
        if deps.predecessors[g].is_empty() {
            ready_at[g] = Some(0);
        }
    }

    let mut scheduled = vec![false; gates.len()];
    let mut scheduled_count: usize = 0;
    let mut gate_cycle = vec![0u32; gates.len()];
    let max_cycles = std::cmp::max(max_depth * 2, gates.len() as u32 + 10);
    let mut cycle = 0u32;

    if timing {
        eprintln!(
            "[sched_emit] list_schedule start: gates={} max_cycles={}",
            gates.len(),
            max_cycles
        );
    }
    while scheduled_count < gates.len() && cycle < max_cycles {
        let mut ready: Vec<usize> = Vec::new();
        for g in 0..gates.len() {
            if !scheduled[g] {
                if let Some(r) = ready_at[g] {
                    if r <= cycle {
                        ready.push(g);
                    }
                }
            }
        }
        ready.sort_by_key(|g| slack[*g]);

        let mut op_counts: HashMap<String, u32> = HashMap::new();
        for g in ready {
            if let Some(max_live) = max_live_pressure {
                let current_live = estimate_live_pressure(
                    cycle,
                    &scheduled,
                    &gate_cycle,
                    &deps,
                    input_bits,
                    outputs,
                );
                if current_live + 1 > max_live {
                    break;
                }
            }
            let op = gate_op_name(&gates[g]).to_string();
            let max_throughput = target.max_per_cycle(&op);
            let current = *op_counts.get(&op).unwrap_or(&0);
            if current < max_throughput {
                gate_cycle[g] = cycle;
                scheduled[g] = true;
                scheduled_count += 1;
                op_counts.insert(op.clone(), current + 1);
                let latency = *latencies.get(&op).unwrap_or(&1);
                let result_ready = cycle + latency;
                for succ in deps.successors[g].iter() {
                    if ready_at[*succ].is_none() {
                        if deps.predecessors[*succ]
                            .iter()
                            .all(|p| scheduled[*p])
                        {
                            let mut pred_ready = 0u32;
                            for p in deps.predecessors[*succ].iter() {
                                let p_cycle = gate_cycle[*p];
                                let p_op = gate_op_name(&gates[*p]).to_string();
                                let p_latency = *latencies.get(&p_op).unwrap_or(&1);
                                pred_ready = pred_ready.max(p_cycle + p_latency);
                            }
                            ready_at[*succ] = Some(pred_ready);
                        }
                    } else {
                        let node_idx = input_bits + g as u32;
                        if gate_operands(&gates[*succ]).contains(&node_idx) {
                            let ready_val = ready_at[*succ].unwrap_or(0).max(result_ready);
                            ready_at[*succ] = Some(ready_val);
                        }
                    }
                }
            }
        }
        cycle += 1;
        if timing && last_report.elapsed().as_secs_f64() >= 5.0 {
            eprintln!(
                "[sched_emit] list_schedule progress: cycle={} scheduled={} ({:.1}%) elapsed={:.1}s",
                cycle,
                scheduled_count,
                (scheduled_count as f64 * 100.0) / (gates.len() as f64),
                t_start.elapsed().as_secs_f64(),
            );
            last_report = Instant::now();
        }
    }

    Schedule { gate_cycle }
}

fn pipelined_schedule(
    gates: &[Gate],
    input_bits: u32,
    outputs: &[(u32, bool)],
    target: &TargetModel,
    timing: bool,
) -> Schedule {
    if gates.is_empty() {
        return Schedule { gate_cycle: vec![] };
    }
    let t_start = Instant::now();
    let mut last_report = Instant::now();
    let deps = compute_dependencies(gates, input_bits);
    let latencies = &target.latencies;
    let num_gates = gates.len();

    let mut scheduled: HashSet<usize> = HashSet::new();
    let mut result_ready_at: HashMap<u32, u32> = HashMap::new();
    for i in 0..input_bits {
        result_ready_at.insert(i, 0);
    }
    let mut ready_at: Vec<i32> = vec![-1; num_gates];
    for g in 0..num_gates {
        if deps.predecessors[g].is_empty() {
            ready_at[g] = 0;
        }
    }

    let mut gate_cycle = vec![0u32; num_gates];
    let max_cycles = num_gates as u32 * 2 + 100;
    let mut cycle = 0u32;

    if timing {
        eprintln!(
            "[sched_emit] pipelined_schedule start: gates={} max_cycles={}",
            num_gates, max_cycles
        );
    }
    while scheduled.len() < num_gates && cycle < max_cycles {
        for g in 0..num_gates {
            if scheduled.contains(&g) {
                continue;
            }
            if ready_at[g] >= 0 {
                continue;
            }
            let mut all_ready = true;
            let mut earliest = 0u32;
            for pred in deps.predecessors[g].iter() {
                let pred_node = input_bits + *pred as u32;
                if let Some(ready) = result_ready_at.get(&pred_node) {
                    earliest = earliest.max(*ready);
                } else {
                    all_ready = false;
                    break;
                }
            }
            if all_ready {
                ready_at[g] = earliest as i32;
            }
        }

        let mut ready: Vec<usize> = Vec::new();
        for g in 0..num_gates {
            if scheduled.contains(&g) {
                continue;
            }
            let r = ready_at[g];
            if r >= 0 && (r as u32) <= cycle {
                ready.push(g);
            }
        }
        ready.sort_by_key(|g| (ready_at[*g], *g as i32));

        let mut op_counts: HashMap<String, u32> = HashMap::new();
        for g in ready {
            let op = gate_op_name(&gates[g]).to_string();
            let max_throughput = target.max_per_cycle(&op);
            let current = *op_counts.get(&op).unwrap_or(&0);
            if current < max_throughput {
                gate_cycle[g] = cycle;
                scheduled.insert(g);
                op_counts.insert(op.clone(), current + 1);
                let latency = *latencies.get(&op).unwrap_or(&1);
                let node_idx = input_bits + g as u32;
                result_ready_at.insert(node_idx, cycle + latency);
            }
        }
        cycle += 1;
        if timing && last_report.elapsed().as_secs_f64() >= 5.0 {
            eprintln!(
                "[sched_emit] pipelined_schedule progress: cycle={} scheduled={} ({:.1}%) elapsed={:.1}s",
                cycle,
                scheduled.len(),
                (scheduled.len() as f64 * 100.0) / (num_gates as f64),
                t_start.elapsed().as_secs_f64(),
            );
            last_report = Instant::now();
        }
    }
    Schedule { gate_cycle }
}

fn estimate_live_pressure(
    cycle: u32,
    scheduled: &[bool],
    gate_cycle: &[u32],
    deps: &DependencyInfo,
    input_bits: u32,
    outputs: &[(u32, bool)],
) -> u32 {
    let mut count = input_bits;
    for (g, is_sched) in scheduled.iter().enumerate() {
        if !*is_sched {
            continue;
        }
        let gate_ready = gate_cycle[g];
        if gate_ready > cycle {
            continue;
        }
        let mut has_future_use = false;
        for succ in deps.successors[g].iter() {
            if !scheduled[*succ] || gate_cycle[*succ] > cycle {
                has_future_use = true;
                break;
            }
        }
        let node_idx = input_bits + g as u32;
        let is_output = outputs.iter().any(|(idx, _)| *idx == node_idx);
        if has_future_use || is_output {
            count += 1;
        }
    }
    count
}

#[derive(Clone, Debug)]
struct LiveRange {
    node: u32,
    start: u32,
    end: u32,
    is_input: bool,
}

fn compute_live_ranges(
    schedule: &Schedule,
    gates: &[Gate],
    input_bits: u32,
    outputs: &[(u32, bool)],
) -> HashMap<u32, LiveRange> {
    if gates.is_empty() && outputs.is_empty() {
        return HashMap::new();
    }
    let total_cycles = schedule.total_cycles();
    let final_cycle = total_cycles.saturating_sub(1);
    let mut last_use: HashMap<u32, u32> = HashMap::new();
    for (g_idx, gate) in gates.iter().enumerate() {
        let cycle = schedule.gate_cycle[g_idx];
        for operand in gate_operands(gate) {
            last_use
                .entry(operand)
                .and_modify(|v| *v = (*v).max(cycle))
                .or_insert(cycle);
        }
    }
    for (out_idx, _) in outputs {
        last_use
            .entry(*out_idx)
            .and_modify(|v| *v = (*v).max(final_cycle))
            .or_insert(final_cycle);
    }
    let mut ranges = HashMap::new();
    for i in 0..input_bits {
        if let Some(end) = last_use.get(&i) {
            ranges.insert(
                i,
                LiveRange {
                    node: i,
                    start: 0,
                    end: *end,
                    is_input: true,
                },
            );
        }
    }
    for g_idx in 0..gates.len() {
        let node_idx = input_bits + g_idx as u32;
        let start = schedule.gate_cycle[g_idx];
        let end = *last_use.get(&node_idx).unwrap_or(&start);
        ranges.insert(
            node_idx,
            LiveRange {
                node: node_idx,
                start,
                end,
                is_input: false,
            },
        );
    }
    ranges
}

#[derive(Clone, Debug)]
struct RegAllocation {
    reg_assignment: HashMap<u32, i32>,
    spills: Vec<u32>,
}

fn allocate_registers(
    live_ranges: &HashMap<u32, LiveRange>,
    schedule: &Schedule,
    num_registers: u32,
    gates: &[Gate],
    input_bits: u32,
    outputs: &[(u32, bool)],
) -> RegAllocation {
    if live_ranges.is_empty() {
        return RegAllocation {
            reg_assignment: HashMap::new(),
            spills: Vec::new(),
        };
    }
    let mut ranges: Vec<LiveRange> = live_ranges.values().cloned().collect();
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut reg_assignment: HashMap<u32, i32> = HashMap::new();
    let mut spills: Vec<u32> = Vec::new();
    let mut active: Vec<LiveRange> = Vec::new();
    let mut free_regs: HashSet<i32> = (0..num_registers as i32).collect();

    let mut uses: HashMap<u32, Vec<u32>> = HashMap::new();
    for (g_idx, gate) in gates.iter().enumerate() {
        let cycle = schedule.gate_cycle[g_idx];
        for operand in gate_operands(gate) {
            uses.entry(operand).or_default().push(cycle);
        }
    }
    let final_cycle = schedule.total_cycles().saturating_sub(1);
    for (out_node, _) in outputs {
        uses.entry(*out_node).or_default().push(final_cycle);
    }
    for vals in uses.values_mut() {
        vals.sort_unstable();
    }

    for current in ranges {
        active.retain(|r| r.end >= current.start);
        let mut used_regs: HashSet<i32> = active
            .iter()
            .filter_map(|r| reg_assignment.get(&r.node).copied())
            .collect();
        free_regs = (0..num_registers as i32)
            .filter(|r| !used_regs.contains(r))
            .collect();

        if let Some(reg) = free_regs.iter().min().copied() {
            free_regs.remove(&reg);
            reg_assignment.insert(current.node, reg);
            active.push(current);
            active.sort_by_key(|r| r.end);
        } else {
            // spill the longest-lived active range
            if let Some(longest_idx) = active
                .iter()
                .enumerate()
                .max_by_key(|(_, r)| r.end)
                .map(|(i, _)| i)
            {
                let longest = active.remove(longest_idx);
                let reg = reg_assignment.get(&longest.node).copied().unwrap_or(-1);
                spills.push(longest.node);
                reg_assignment.insert(current.node, reg);
                active.push(current);
                active.sort_by_key(|r| r.end);
            } else {
                spills.push(current.node);
            }
        }
    }

    RegAllocation {
        reg_assignment,
        spills,
    }
}

fn needs_ones_constant(gates: &[Gate], outputs: &[(u32, bool)]) -> bool {
    for g in gates {
        match g {
            Gate::Unary { op, .. } if op == "not" => return true,
            Gate::Const { .. } => return true,
            _ => {}
        }
    }
    if outputs.iter().any(|(_, inv)| *inv) {
        return true;
    }
    false
}

fn emit_naive(
    circuit: &Circuit,
    function_name: &str,
    io_split: Option<(u32, u32)>,
    ops: &EmitOps,
) -> String {
    let input_bits = circuit.input_bits as usize;
    let output_len = circuit.outputs.len();
    let (input_io_bits, output_io_bits) = io_split.unwrap_or((input_bits as u32, output_len as u32));
    let state_bits = input_bits as i32 - input_io_bits as i32;
    let mut lines: Vec<String> = Vec::new();
    lines.push("#include <immintrin.h>".to_string());
    lines.push(String::new());
    if io_split.is_none() {
        lines.push(format!(
            "void {function_name}({}* in, {}* out) {{",
            ops.vec_ty, ops.vec_ty
        ));
    } else {
        lines.push(format!(
            "static inline void {function_name}__core(const {ty}* in_io, const {ty}* st_in, {ty}* out_io, {ty}* st_out) {{",
            ty = ops.vec_ty
        ));
    }
    let mut used_regs: HashSet<u32> = (0..circuit.input_bits).collect();
    for (g_idx, gate) in circuit.gates.iter().enumerate() {
        let dst = circuit.input_bits + g_idx as u32;
        used_regs.insert(dst);
        for op in gate_operands(gate) {
            used_regs.insert(op);
        }
    }
    for (node_idx, _) in circuit.outputs.iter() {
        used_regs.insert(*node_idx);
    }
    for r in used_regs.iter().copied().collect::<Vec<u32>>() {
        lines.push(format!("    {} r{r};", ops.vec_ty));
    }
    if needs_ones_constant(&circuit.gates, &circuit.outputs) {
        lines.push(format!("    {} ones = {};", ops.vec_ty, ops.set1));
    }
    for i in 0..input_bits {
        if io_split.is_none() {
            lines.push(format!("    r{i} = in[{i}];"));
        } else if i < input_io_bits as usize {
            lines.push(format!("    r{i} = in_io[{i}];"));
        } else {
            lines.push(format!(
                "    r{i} = st_in[{}];",
                i - input_io_bits as usize
            ));
        }
    }
    for (g_idx, gate) in circuit.gates.iter().enumerate() {
        let dst = circuit.input_bits + g_idx as u32;
        let line = emit_gate_line(gate, dst, true, ops);
        lines.push(format!("    {line}"));
    }
    for (out_idx, (node_idx, inverted)) in circuit.outputs.iter().enumerate() {
        if io_split.is_none() {
            if *inverted {
                lines.push(format!(
                    "    out[{out_idx}] = {}(r{node_idx}, ones);",
                    ops.xor
                ));
            } else {
                lines.push(format!("    out[{out_idx}] = r{node_idx};"));
            }
        } else {
            let dst = if (out_idx as u32) < output_io_bits {
                format!("out_io[{out_idx}]")
            } else {
                format!("st_out[{}]", out_idx as u32 - output_io_bits)
            };
            if *inverted {
                lines.push(format!("    {dst} = {}(r{node_idx}, ones);", ops.xor));
            } else {
                lines.push(format!("    {dst} = r{node_idx};"));
            }
        }
    }
    lines.push("}".to_string());
    lines.push(String::new());
    if io_split.is_some() {
        lines.push(format!(
            "void {function_name}({}* in, {}* out) {{",
            ops.vec_ty, ops.vec_ty
        ));
        lines.push(format!(
            "    {function_name}__core(in, in + {input_io_bits}, out, out + {output_io_bits});"
        ));
        lines.push("}".to_string());
        lines.push(String::new());
        lines.push(format!(
            "void {function_name}_steps_shared(const {ty}* in_io, {ty}* out_io, const {ty}* state_in, {ty}* state_out, int steps) {{",
            ty = ops.vec_ty
        ));
        lines.push(format!("    const {}* st_r = state_in;", ops.vec_ty));
        lines.push(format!("    {}* st_w = state_out;", ops.vec_ty));
        lines.push("    for (int k = 0; k < steps; k++) {".to_string());
        lines.push(format!(
            "        {function_name}__core(in_io, st_r, out_io, st_w);"
        ));
        lines.push(format!("        const {}* tmp = st_r;", ops.vec_ty));
        lines.push("        st_r = st_w;".to_string());
        lines.push(format!("        st_w = ({ty}*)tmp;", ty = ops.vec_ty));
        lines.push("    }".to_string());
        lines.push("    if ((steps & 1) == 0) {".to_string());
        lines.push(format!(
            "        for (int i = 0; i < {state_bits}; i++) state_out[i] = state_in[i];"
        ));
        lines.push("    }".to_string());
        lines.push("}".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

fn emit_gate_line(gate: &Gate, dst: u32, use_regs: bool, ops: &EmitOps) -> String {
    let r = |idx: u32| {
        if use_regs {
            format!("r{idx}")
        } else {
            format!("r{idx}")
        }
    };
    match gate {
        Gate::Ternary { a, b, c, imm8 } => format!(
            "r{dst} = {}({}, {}, {}, {});",
            ops.ternary,
            r(*a),
            r(*b),
            r(*c),
            imm8
        ),
        Gate::Binary { op, a, b } if op == "andn" || op == "andnot" => {
            format!("r{dst} = {}({}, {});", ops.andnot, r(*a), r(*b))
        }
        Gate::Binary { op, a, b } if op == "xor" => {
            format!("r{dst} = {}({}, {});", ops.xor, r(*a), r(*b))
        }
        Gate::Binary { op, a, b } if op == "and" => {
            format!("r{dst} = {}({}, {});", ops.and, r(*a), r(*b))
        }
        Gate::Binary { op, a, b } if op == "or" => {
            format!("r{dst} = {}({}, {});", ops.or, r(*a), r(*b))
        }
        Gate::Unary { op, a } if op == "not" => {
            format!("r{dst} = {}({}, ones);", ops.xor, r(*a))
        }
        Gate::Const { value, .. } => {
            if *value == 0 {
                format!("r{dst} = {};", ops.setzero)
            } else {
                format!("r{dst} = {};", ops.set1)
            }
        }
        _ => format!("r{dst} = {};", ops.setzero),
    }
}

fn emit_scheduled(
    circuit: &Circuit,
    schedule: &Schedule,
    allocation: &RegAllocation,
    function_name: &str,
    io_split: Option<(u32, u32)>,
    ops: &EmitOps,
) -> String {
    if !allocation.spills.is_empty() {
        return emit_naive(circuit, function_name, io_split, ops);
    }
    let input_bits = circuit.input_bits as usize;
    let output_len = circuit.outputs.len();
    let (input_io_bits, output_io_bits) = io_split.unwrap_or((input_bits as u32, output_len as u32));
    let state_bits = input_bits as i32 - input_io_bits as i32;

    let mut lines: Vec<String> = Vec::new();
    lines.push("#include <immintrin.h>".to_string());
    lines.push(String::new());
    if io_split.is_none() {
        lines.push(format!(
            "void {function_name}({}* in, {}* out) {{",
            ops.vec_ty, ops.vec_ty
        ));
    } else {
        lines.push(format!(
            "static inline void {function_name}__core(const {ty}* in_io, const {ty}* st_in, {ty}* out_io, {ty}* st_out) {{",
            ty = ops.vec_ty
        ));
    }

    let mut used_regs: HashSet<i32> = allocation.reg_assignment.values().copied().collect();
    let max_reg = used_regs.iter().copied().max().unwrap_or(0).max(input_bits as i32 - 1);
    for r in 0..=max_reg {
        lines.push(format!("    {} r{r};", ops.vec_ty));
    }
    if needs_ones_constant(&circuit.gates, &circuit.outputs) {
        lines.push(format!("    {} ones = {};", ops.vec_ty, ops.set1));
    }
    for i in 0..input_bits {
        if io_split.is_none() {
            lines.push(format!("    r{i} = in[{i}];"));
        } else if i < input_io_bits as usize {
            lines.push(format!("    r{i} = in_io[{i}];"));
        } else {
            lines.push(format!(
                "    r{i} = st_in[{}];",
                i - input_io_bits as usize
            ));
        }
    }

    let mut gates_by_cycle: HashMap<u32, Vec<usize>> = HashMap::new();
    for (g_idx, cycle) in schedule.gate_cycle.iter().enumerate() {
        gates_by_cycle.entry(*cycle).or_default().push(g_idx);
    }
    let mut node_in_reg: HashMap<u32, i32> = HashMap::new();
    let mut reg_holds_node: HashMap<i32, u32> = HashMap::new();
    for i in 0..input_bits {
        node_in_reg.insert(i as u32, i as i32);
        reg_holds_node.insert(i as i32, i as u32);
    }

    let total_cycles = schedule.total_cycles();
    for cycle in 0..total_cycles {
        if let Some(gates) = gates_by_cycle.get(&cycle) {
            for g_idx in gates {
                let node_idx = circuit.input_bits + *g_idx as u32;
                let dst_reg = *allocation.reg_assignment.get(&node_idx).unwrap_or(&-1);
                if dst_reg < 0 {
                    continue;
                }
                let line = emit_gate_line_scheduled(
                    &circuit.gates[*g_idx],
                    dst_reg,
                    &node_in_reg,
                    &allocation.reg_assignment,
                    ops,
                );
                lines.push(format!("    {line}"));
                if let Some(old_node) = reg_holds_node.get(&dst_reg).copied() {
                    node_in_reg.remove(&old_node);
                }
                node_in_reg.insert(node_idx, dst_reg);
                reg_holds_node.insert(dst_reg, node_idx);
            }
        }
    }

    for (out_idx, (node_idx, inverted)) in circuit.outputs.iter().enumerate() {
        let reg = node_in_reg
            .get(node_idx)
            .copied()
            .or_else(|| allocation.reg_assignment.get(node_idx).copied())
            .unwrap_or(*node_idx as i32);
        if io_split.is_none() {
            if *inverted {
                lines.push(format!(
                    "    out[{out_idx}] = {}(r{reg}, ones);",
                    ops.xor
                ));
            } else {
                lines.push(format!("    out[{out_idx}] = r{reg};"));
            }
        } else {
            let dst = if (out_idx as u32) < output_io_bits {
                format!("out_io[{out_idx}]")
            } else {
                format!("st_out[{}]", out_idx as u32 - output_io_bits)
            };
            if *inverted {
                lines.push(format!("    {dst} = {}(r{reg}, ones);", ops.xor));
            } else {
                lines.push(format!("    {dst} = r{reg};"));
            }
        }
    }
    lines.push("}".to_string());
    lines.push(String::new());

    if io_split.is_some() {
        lines.push(format!(
            "void {function_name}({}* in, {}* out) {{",
            ops.vec_ty, ops.vec_ty
        ));
        lines.push(format!(
            "    {function_name}__core(in, in + {input_io_bits}, out, out + {output_io_bits});"
        ));
        lines.push("}".to_string());
        lines.push(String::new());
        lines.push(format!(
            "void {function_name}_steps_shared(const {ty}* in_io, {ty}* out_io, const {ty}* state_in, {ty}* state_out, int steps) {{",
            ty = ops.vec_ty
        ));
        lines.push(format!("    const {}* st_r = state_in;", ops.vec_ty));
        lines.push(format!("    {}* st_w = state_out;", ops.vec_ty));
        lines.push("    for (int k = 0; k < steps; k++) {".to_string());
        lines.push(format!(
            "        {function_name}__core(in_io, st_r, out_io, st_w);"
        ));
        lines.push(format!("        const {}* tmp = st_r;", ops.vec_ty));
        lines.push("        st_r = st_w;".to_string());
        lines.push(format!("        st_w = ({ty}*)tmp;", ty = ops.vec_ty));
        lines.push("    }".to_string());
        lines.push("    if ((steps & 1) == 0) {".to_string());
        lines.push(format!(
            "        for (int i = 0; i < {state_bits}; i++) state_out[i] = state_in[i];"
        ));
        lines.push("    }".to_string());
        lines.push("}".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

fn emit_gate_line_scheduled(
    gate: &Gate,
    dst_reg: i32,
    node_in_reg: &HashMap<u32, i32>,
    reg_assignment: &HashMap<u32, i32>,
    ops: &EmitOps,
) -> String {
    let node_expr = |node: u32| -> String {
        if let Some(reg) = node_in_reg.get(&node) {
            format!("r{reg}")
        } else if let Some(reg) = reg_assignment.get(&node) {
            format!("r{reg}")
        } else {
            format!("r{node}")
        }
    };
    match gate {
        Gate::Ternary { a, b, c, imm8 } => format!(
            "r{dst_reg} = {}({}, {}, {}, {});",
            ops.ternary,
            node_expr(*a),
            node_expr(*b),
            node_expr(*c),
            imm8
        ),
        Gate::Binary { op, a, b } if op == "andn" || op == "andnot" => {
            format!(
                "r{dst_reg} = {}({}, {});",
                ops.andnot,
                node_expr(*a),
                node_expr(*b)
            )
        }
        Gate::Binary { op, a, b } if op == "xor" => {
            format!("r{dst_reg} = {}({}, {});", ops.xor, node_expr(*a), node_expr(*b))
        }
        Gate::Binary { op, a, b } if op == "and" => {
            format!("r{dst_reg} = {}({}, {});", ops.and, node_expr(*a), node_expr(*b))
        }
        Gate::Binary { op, a, b } if op == "or" => {
            format!("r{dst_reg} = {}({}, {});", ops.or, node_expr(*a), node_expr(*b))
        }
        Gate::Unary { op, a } if op == "not" => {
            format!("r{dst_reg} = {}({}, ones);", ops.xor, node_expr(*a))
        }
        Gate::Const { value, .. } => {
            if *value == 0 {
                format!("r{dst_reg} = {};", ops.setzero)
            } else {
                format!("r{dst_reg} = {};", ops.set1)
            }
        }
        _ => format!("r{dst_reg} = {};", ops.setzero),
    }
}

fn serial_schedule(gates: &[Gate]) -> Schedule {
    Schedule {
        gate_cycle: (0..gates.len()).map(|g| g as u32).collect(),
    }
}

fn has_multi_per_cycle(schedule: &Schedule) -> bool {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for cycle in schedule.gate_cycle.iter() {
        *counts.entry(*cycle).or_default() += 1;
        if counts[cycle] > 1 {
            return true;
        }
    }
    false
}

fn main() -> Result<()> {
    let args = Args::parse();
    let timing = args.timing;
    let t0_total = Instant::now();
    let circuit = if args.format == "bin" {
        let t0 = Instant::now();
        let out = parse_circuit_bin(&args.input)?;
        if timing {
            eprintln!("[sched_emit] parse_circuit_bin: {:.3}s", t0.elapsed().as_secs_f64());
        }
        out
    } else {
        let t0 = Instant::now();
        let out = parse_circuit_json(&args.input)?;
        if timing {
            eprintln!("[sched_emit] parse_circuit_json: {:.3}s", t0.elapsed().as_secs_f64());
        }
        out
    };
    let target = match args.target.as_str() {
        "avx512" => target_avx512(),
        "avx2" => target_avx2(),
        other => bail!("unsupported target {}", other),
    };
    let ops = emit_ops(&args.target)?;
    let io_split = args
        .input_io_bits
        .zip(args.output_io_bits)
        .map(|(a, b)| (a, b));

    let t0 = Instant::now();
    let schedule = match args.scheduler.as_str() {
        "list" => list_schedule(
            &circuit.gates,
            circuit.input_bits,
            &circuit.outputs,
            &target,
            args.max_live_pressure,
            timing,
        ),
        "pipelined" => pipelined_schedule(
            &circuit.gates,
            circuit.input_bits,
            &circuit.outputs,
            &target,
            timing,
        ),
        "serial" => {
            if timing {
                eprintln!(
                    "[sched_emit] serial_schedule start: gates={}",
                    circuit.gates.len()
                );
            }
            serial_schedule(&circuit.gates)
        }
        other => bail!("unsupported scheduler {}", other),
    };
    if timing {
        eprintln!("[sched_emit] schedule: {:.3}s", t0.elapsed().as_secs_f64());
    }
    let t0 = Instant::now();
    let live_ranges = compute_live_ranges(
        &schedule,
        &circuit.gates,
        circuit.input_bits,
        &circuit.outputs,
    );
    if timing {
        eprintln!(
            "[sched_emit] compute_live_ranges: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    let t0 = Instant::now();
    let mut allocation = allocate_registers(
        &live_ranges,
        &schedule,
        target.registers,
        &circuit.gates,
        circuit.input_bits,
        &circuit.outputs,
    );
    if timing {
        eprintln!(
            "[sched_emit] allocate_registers: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    let mut final_schedule = schedule;
    if !allocation.spills.is_empty() && has_multi_per_cycle(&final_schedule) {
        let t0 = Instant::now();
        let serial = serial_schedule(&circuit.gates);
        let live_ranges = compute_live_ranges(
            &serial,
            &circuit.gates,
            circuit.input_bits,
            &circuit.outputs,
        );
        if timing {
            eprintln!(
                "[sched_emit] serial compute_live_ranges: {:.3}s",
                t0.elapsed().as_secs_f64()
            );
        }
        let t1 = Instant::now();
        allocation = allocate_registers(
            &live_ranges,
            &serial,
            target.registers,
            &circuit.gates,
            circuit.input_bits,
            &circuit.outputs,
        );
        if timing {
            eprintln!(
                "[sched_emit] serial allocate_registers: {:.3}s",
                t1.elapsed().as_secs_f64()
            );
        }
        final_schedule = serial;
        if timing {
            eprintln!(
                "[sched_emit] serialization path total: {:.3}s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let t0 = Instant::now();
    let code = emit_scheduled(
        &circuit,
        &final_schedule,
        &allocation,
        &args.function_name,
        io_split,
        &ops,
    );
    if timing {
        eprintln!("[sched_emit] emit_scheduled: {:.3}s", t0.elapsed().as_secs_f64());
    }
    fs::write(&args.output, code)?;
    if timing {
        eprintln!("[sched_emit] total: {:.3}s", t0_total.elapsed().as_secs_f64());
    }
    Ok(())
}
