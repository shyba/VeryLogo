use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use num_bigint::BigUint;
use num_traits::Zero;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
enum Ty {
    Bool,
    BitVec { width: u64 },
    Float { width: u64 },
    Simd { lane_width: u64, lanes: u64 },
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "bin")]
    format: String,
    #[arg(long, default_value = "json")]
    output_format: String,
}

#[derive(Clone, Debug)]
enum Node {
    Var { name_idx: u32 },
    BoolConst(bool),
    BitVecConst { width: u32, value: BigUint },
    FloatConst { width: u32, bits: u64 },
    SimdConst { lane_width: u32, lanes: u32, value: BigUint },
    Unary { op: u8, x: u32 },
    Binary { op: u8, a: u32, b: u32 },
    Ternary { op: u8, a: u32, b: u32, c: u32 },
    Mux { cond: u32, a: u32, b: u32 },
    Concat { parts: Vec<u32> },
    Slice { x: u32, offset: u32, width: u32 },
    Lut8 { x: u32, table: Vec<u8> },
    TernaryLut { a: u32, b: u32, c: u32, imm8: u8 },
    Bitcast { to: TypeVal, x: u32 },
    BitTranspose { x: u32, lane_width: u32, lanes: u32 },
    Rot { op: u8, x: u32, sh: u32 },
    Delay { x: u32, ticks: u32 },
    Masked { op: u8, mask: u32, a: u32, b: u32 },
    MaskExpand { to: TypeVal, x: u32 },
    MaskPack { x: u32 },
    Ext { op: u8, to: TypeVal, x: u32 },
    Splat { to: TypeVal, x: u32 },
    ExtractLane { x: u32, lane: u32 },
    InsertLane { x: u32, lane: u32, value: u32 },
    Shuffle { x: u32, indices: Vec<u32> },
}

#[derive(Clone, Debug)]
enum TypeVal {
    Bool,
    BitVec { width: u32 },
    Float { width: u32 },
    Simd { lane_width: u32, lanes: u32 },
}

#[derive(Serialize)]
struct CircuitStateOut {
    input_bits: u32,
    output_bits: u32,
    gates: Vec<Value>,
    outputs: Vec<(u32, bool)>,
    gate_count: usize,
}

#[derive(Clone, Debug)]
enum Gate {
    Const { value: u32, width: u32 },
    Unary { op: String, a: u32 },
    Binary { op: String, a: u32, b: u32 },
    Ternary { a: u32, b: u32, c: u32, imm8: u8 },
}

impl Gate {
    fn op_name(&self) -> &str {
        match self {
            Gate::Const { .. } => "const",
            Gate::Unary { op, .. } => op,
            Gate::Binary { op, .. } => op,
            Gate::Ternary { .. } => "ternary",
        }
    }

    fn to_value(&self) -> Value {
        match self {
            Gate::Const { value, width } => Value::Array(vec![
                Value::String("const".to_string()),
                Value::Number((*value).into()),
                Value::Number((*width).into()),
            ]),
            Gate::Unary { op, a } => Value::Array(vec![
                Value::String(op.clone()),
                Value::Number((*a).into()),
                Value::Number(0u32.into()),
            ]),
            Gate::Binary { op, a, b } => Value::Array(vec![
                Value::String(op.clone()),
                Value::Number((*a).into()),
                Value::Number((*b).into()),
            ]),
            Gate::Ternary { a, b, c, imm8 } => Value::Array(vec![
                Value::String("ternary".to_string()),
                Value::Number((*a).into()),
                Value::Number((*b).into()),
                Value::Number((*c).into()),
                Value::Number((*imm8 as u32).into()),
            ]),
        }
    }
}

#[derive(Clone, Debug)]
struct Circuit {
    input_bits: u32,
    output_bits: u32,
    gates: Vec<Gate>,
    outputs: Vec<(u32, bool)>,
}

struct LayoutEntry {
    lsb: u32,
    width: u32,
}

struct PackedLayout {
    inputs: HashMap<String, LayoutEntry>,
    state: HashMap<String, LayoutEntry>,
    outputs: HashMap<String, LayoutEntry>,
    next_state: HashMap<String, LayoutEntry>,
    input_bits: u32,
    output_bits: u32,
}

fn parse_gates(values: &[Value]) -> Result<Vec<Gate>> {
    let mut gates = Vec::with_capacity(values.len());
    for v in values {
        let arr = v
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("gate is not array"))?;
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
            let imm8 =
                arr[4].as_u64().ok_or_else(|| anyhow::anyhow!("gate imm8"))? as u8;
            gates.push(Gate::Ternary { a, b, c, imm8 });
        } else if arr.len() == 3 {
            let op = arr[0]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("gate op"))?
                .to_string();
            let a = arr[1].as_u64().ok_or_else(|| anyhow::anyhow!("gate a"))? as u32;
            let b = arr[2].as_u64().ok_or_else(|| anyhow::anyhow!("gate b"))? as u32;
            if op == "const" {
                gates.push(Gate::Const { value: a, width: b });
            } else if op == "not" {
                gates.push(Gate::Unary { op, a });
            } else {
                gates.push(Gate::Binary { op, a, b });
            }
        } else {
            bail!("unexpected gate length {}", arr.len());
        }
    }
    Ok(gates)
}

fn gates_to_values(gates: &[Gate]) -> Vec<Value> {
    gates.iter().map(|g| g.to_value()).collect()
}

fn is_truthy_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v != "0" && v != "false" && v != "no")
        .unwrap_or(false)
}

fn const_value(
    circuit: &Circuit,
    signal_idx: u32,
    cache: &mut HashMap<u32, Option<u8>>,
) -> Option<u8> {
    if let Some(v) = cache.get(&signal_idx) {
        return *v;
    }
    if signal_idx < circuit.input_bits {
        cache.insert(signal_idx, None);
        return None;
    }
    let gate_idx = signal_idx as i64 - circuit.input_bits as i64;
    if gate_idx < 0 || gate_idx as usize >= circuit.gates.len() {
        cache.insert(signal_idx, None);
        return None;
    }
    let gate = &circuit.gates[gate_idx as usize];
    let result = match gate {
        Gate::Const { value, .. } => Some((*value & 1) as u8),
        Gate::Unary { op, a } if op == "not" => const_value(circuit, *a, cache).map(|v| v ^ 1),
        Gate::Binary { op, a, b } if op == "xor" => {
            let va = const_value(circuit, *a, cache)?;
            let vb = const_value(circuit, *b, cache)?;
            Some(va ^ vb)
        }
        Gate::Binary { op, a, b } if op == "and" => {
            let va = const_value(circuit, *a, cache)?;
            let vb = const_value(circuit, *b, cache)?;
            Some(va & vb)
        }
        Gate::Binary { op, a, b } if op == "or" => {
            let va = const_value(circuit, *a, cache)?;
            let vb = const_value(circuit, *b, cache)?;
            Some(va | vb)
        }
        _ => None,
    };
    cache.insert(signal_idx, result);
    result
}

fn symmetric_diff(a: &HashSet<u32>, b: &HashSet<u32>) -> HashSet<u32> {
    let mut out = HashSet::new();
    for v in a {
        if !b.contains(v) {
            out.insert(*v);
        }
    }
    for v in b {
        if !a.contains(v) {
            out.insert(*v);
        }
    }
    out
}

fn collect_linear_deps(
    circuit: &Circuit,
    signal_idx: u32,
    stop_at: &HashSet<u32>,
    cone_inputs: &mut HashSet<u32>,
    const_cache: &mut HashMap<u32, Option<u8>>,
    memo: &mut Vec<Option<HashSet<u32>>>,
) -> HashSet<u32> {
    if let Some(entry) = memo.get(signal_idx as usize).and_then(|v| v.clone()) {
        cone_inputs.extend(entry.iter().copied());
        return entry;
    }
    if stop_at.contains(&signal_idx) || signal_idx < circuit.input_bits {
        cone_inputs.insert(signal_idx);
        let mut set = HashSet::new();
        set.insert(signal_idx);
        memo[signal_idx as usize] = Some(set.clone());
        return set;
    }
    let gate_idx = signal_idx as i64 - circuit.input_bits as i64;
    if gate_idx < 0 || gate_idx as usize >= circuit.gates.len() {
        cone_inputs.insert(signal_idx);
        let mut set = HashSet::new();
        set.insert(signal_idx);
        memo[signal_idx as usize] = Some(set.clone());
        return set;
    }
    let gate = &circuit.gates[gate_idx as usize];
    let result = match gate {
        Gate::Ternary { .. } => {
            let mut set = HashSet::new();
            set.insert(signal_idx);
            set
        }
        Gate::Unary { op, a } if op == "not" => {
            collect_linear_deps(circuit, *a, stop_at, cone_inputs, const_cache, memo)
        }
        Gate::Binary { op, a, b } if op == "xor" => {
            let left = collect_linear_deps(circuit, *a, stop_at, cone_inputs, const_cache, memo);
            let right = collect_linear_deps(circuit, *b, stop_at, cone_inputs, const_cache, memo);
            symmetric_diff(&left, &right)
        }
        Gate::Binary { op, a, b } if op == "and" => {
            let left_const = const_value(circuit, *a, const_cache);
            let right_const = const_value(circuit, *b, const_cache);
            if left_const == Some(0) || right_const == Some(0) {
                HashSet::new()
            } else if left_const == Some(1) {
                collect_linear_deps(circuit, *b, stop_at, cone_inputs, const_cache, memo)
            } else if right_const == Some(1) {
                collect_linear_deps(circuit, *a, stop_at, cone_inputs, const_cache, memo)
            } else if left_const.is_some() && right_const.is_some() {
                HashSet::new()
            } else {
                let mut set = HashSet::new();
                set.insert(signal_idx);
                set
            }
        }
        _ => {
            let mut set = HashSet::new();
            set.insert(signal_idx);
            set
        }
    };
    cone_inputs.extend(result.iter().copied());
    if let Some(slot) = memo.get_mut(signal_idx as usize) {
        *slot = Some(result.clone());
    }
    result
}

#[derive(Clone, Debug)]
struct LinearCone {
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    matrix64: Option<Vec<u64>>,
    row_positions: Vec<Vec<usize>>,
}

impl LinearCone {
    fn from_circuit(
        circuit: &Circuit,
        output_indices: &[u32],
        stop_at: &HashSet<u32>,
    ) -> Self {
        let mut cone_inputs_set: HashSet<u32> = HashSet::new();
        let mut output_dependencies: Vec<HashSet<u32>> = Vec::new();
        let mut const_cache: HashMap<u32, Option<u8>> = HashMap::new();
        let mut memo: Vec<Option<HashSet<u32>>> =
            vec![None; circuit.input_bits as usize + circuit.gates.len() + 1];

        for &out_idx in output_indices {
            let deps = collect_linear_deps(
                circuit,
                out_idx,
                stop_at,
                &mut cone_inputs_set,
                &mut const_cache,
                &mut memo,
            );
            output_dependencies.push(deps);
        }

        let mut inputs: Vec<u32> = cone_inputs_set.into_iter().collect();
        inputs.sort_unstable();
        let mut input_pos: HashMap<u32, usize> = HashMap::new();
        for (i, inp) in inputs.iter().enumerate() {
            input_pos.insert(*inp, i);
        }

        let mut row_positions: Vec<Vec<usize>> = Vec::new();
        let mut matrix64: Option<Vec<u64>> = None;
        if inputs.len() <= 64 {
            let mut matrix = Vec::with_capacity(output_dependencies.len());
            for deps in &output_dependencies {
                let mut row = 0u64;
                let mut positions = Vec::new();
                for dep in deps {
                    if let Some(pos) = input_pos.get(dep) {
                        row |= 1u64 << pos;
                        positions.push(*pos);
                    }
                }
                positions.sort_unstable();
                row_positions.push(positions);
                matrix.push(row);
            }
            matrix64 = Some(matrix);
        } else {
            for deps in &output_dependencies {
                let mut positions = Vec::new();
                for dep in deps {
                    if let Some(pos) = input_pos.get(dep) {
                        positions.push(*pos);
                    }
                }
                positions.sort_unstable();
                row_positions.push(positions);
            }
        }

        LinearCone {
            inputs,
            outputs: output_indices.to_vec(),
            matrix64,
            row_positions,
        }
    }

    fn to_xor_circuit_optimized(&self, max_inputs_for_bp: usize) -> Vec<(usize, usize)> {
        let num_inputs = self.inputs.len();
        if num_inputs == 0 {
            return Vec::new();
        }
        if let Some(matrix) = &self.matrix64 {
            if num_inputs <= max_inputs_for_bp {
                return boyar_peralta_minimize(matrix, num_inputs);
            }
        }
        self.to_xor_circuit()
    }

    fn to_xor_circuit(&self) -> Vec<(usize, usize)> {
        let num_inputs = self.inputs.len();
        let mut gates: Vec<(usize, usize)> = Vec::new();
        for row in &self.row_positions {
            if row.len() <= 1 {
                continue;
            }
            let mut current = row[0];
            for next in row.iter().skip(1) {
                let gate_idx = num_inputs + gates.len();
                gates.push((current, *next));
                current = gate_idx;
            }
        }
        gates
    }

    fn get_output_signals_optimized(&self, max_inputs_for_bp: usize) -> Vec<i32> {
        let num_inputs = self.inputs.len();
        if num_inputs == 0 {
            return vec![];
        }
        if let Some(matrix) = &self.matrix64 {
            if num_inputs <= max_inputs_for_bp {
                return get_output_signals_bp(matrix, num_inputs);
            }
        }
        self.get_output_signals_naive()
    }

    fn get_output_signals_naive(&self) -> Vec<i32> {
        let num_inputs = self.inputs.len();
        let mut output_signals: Vec<i32> = Vec::new();
        let mut gate_count = 0usize;
        for row in &self.row_positions {
            if row.is_empty() {
                output_signals.push(-1);
                continue;
            }
            if row.len() == 1 {
                output_signals.push(row[0] as i32);
                continue;
            }
            gate_count += row.len() - 1;
            output_signals.push((num_inputs + gate_count - 1) as i32);
        }
        output_signals
    }
}

fn boyar_peralta_minimize(matrix: &[u64], n_inputs: usize) -> Vec<(usize, usize)> {
    let mut base: Vec<u64> = (0..n_inputs).map(|i| 1u64 << i).collect();
    let mut base_set: HashSet<u64> = base.iter().copied().collect();
    let mut targets: HashSet<u64> = matrix.iter().copied().collect();
    let mut remaining: HashSet<u64> = targets.drain().filter(|t| !base_set.contains(t)).collect();
    let mut operations: Vec<(usize, usize)> = Vec::new();

    while !remaining.is_empty() {
        let mut best_pair: Option<(usize, usize)> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_candidate = 0u64;

        for i in 0..base.len() {
            for j in (i + 1)..base.len() {
                let candidate = base[i] ^ base[j];
                if base_set.contains(&candidate) {
                    continue;
                }
                let mut score = 0.0;
                for t in &remaining {
                    let mut old_dist = 64;
                    for b in &base {
                        let dist = (t ^ b).count_ones() as i32;
                        if dist < old_dist {
                            old_dist = dist;
                        }
                    }
                    let mut new_dist = old_dist;
                    let dist = (t ^ candidate).count_ones() as i32;
                    if dist < new_dist {
                        new_dist = dist;
                    }
                    score += (old_dist - new_dist) as f64;
                }
                if remaining.contains(&candidate) {
                    score += remaining.len() as f64 * 10.0;
                }
                if score > best_score {
                    best_score = score;
                    best_pair = Some((i, j));
                    best_candidate = candidate;
                }
            }
        }
        let (a, b) = best_pair.expect("no bp pair");
        operations.push((a, b));
        base.push(best_candidate);
        base_set.insert(best_candidate);
        remaining.remove(&best_candidate);
    }
    operations
}

fn get_output_signals_bp(matrix: &[u64], n_inputs: usize) -> Vec<i32> {
    let mut base: Vec<u64> = (0..n_inputs).map(|i| 1u64 << i).collect();
    let mut base_set: HashSet<u64> = base.iter().copied().collect();
    let non_trivial: Vec<u64> = matrix.iter().copied().filter(|row| row.count_ones() > 1).collect();
    if !non_trivial.is_empty() {
        let ops = boyar_peralta_minimize(&non_trivial, n_inputs);
        for (a, b) in ops {
            let candidate = base[a] ^ base[b];
            if !base_set.contains(&candidate) {
                base.push(candidate);
                base_set.insert(candidate);
            }
        }
    }

    let mut row_to_signal: HashMap<u64, i32> = HashMap::new();
    for i in 0..n_inputs {
        row_to_signal.insert(1u64 << i, i as i32);
    }
    for (i, val) in base[n_inputs..].iter().enumerate() {
        row_to_signal.insert(*val, (n_inputs + i) as i32);
    }

    let mut output_signals = Vec::with_capacity(matrix.len());
    for row in matrix {
        if *row == 0 {
            output_signals.push(-1);
        } else if let Some(sig) = row_to_signal.get(row) {
            output_signals.push(*sig);
        } else {
            output_signals.push(-1);
        }
    }
    output_signals
}

fn optimize_linear_layers(circuit: &Circuit) -> Circuit {
    let mut and_gate_indices: Vec<u32> = Vec::new();
    let mut and_only_indices: Vec<u32> = Vec::new();
    for (g_idx, gate) in circuit.gates.iter().enumerate() {
        let signal = circuit.input_bits + g_idx as u32;
        match gate {
            Gate::Ternary { .. } => and_gate_indices.push(signal),
            Gate::Binary { op, .. } if op == "and" => {
                and_gate_indices.push(signal);
                and_only_indices.push(signal);
            }
            _ => {}
        }
    }

    if and_gate_indices.is_empty() {
        let mut all_xors = Vec::new();
        for (g_idx, gate) in circuit.gates.iter().enumerate() {
            if let Gate::Binary { op, .. } = gate {
                if op == "xor" {
                    all_xors.push(circuit.input_bits + g_idx as u32);
                }
            }
        }
        if all_xors.is_empty() {
            return circuit.clone();
        }
        let mut output_xors = Vec::new();
        for (idx, _) in &circuit.outputs {
            if *idx >= circuit.input_bits {
                let gate_idx = *idx - circuit.input_bits;
                if let Some(Gate::Binary { op, .. }) = circuit.gates.get(gate_idx as usize) {
                    if op == "xor" {
                        output_xors.push(*idx);
                    }
                }
            }
        }
        if output_xors.is_empty() {
            return circuit.clone();
        }

        let stop_at = HashSet::new();
        let cone = LinearCone::from_circuit(circuit, &output_xors, &stop_at);
        let optimized_gates = cone.to_xor_circuit_optimized(64);
        let optimized_output_signals = cone.get_output_signals_optimized(64);

        let mut new_gates: Vec<Gate> = Vec::new();
        let mut old_to_new: HashMap<u32, u32> = HashMap::new();
        for i in 0..circuit.input_bits {
            old_to_new.insert(i, i);
        }

        let mut opt_gate_to_new: HashMap<usize, u32> = HashMap::new();
        for (local_idx, (left, right)) in optimized_gates.iter().enumerate() {
            let new_left = if *left < cone.inputs.len() {
                cone.inputs[*left]
            } else {
                *opt_gate_to_new.get(left).unwrap()
            };
            let new_right = if *right < cone.inputs.len() {
                cone.inputs[*right]
            } else {
                *opt_gate_to_new.get(right).unwrap()
            };
            let new_idx = circuit.input_bits + new_gates.len() as u32;
            new_gates.push(Gate::Binary {
                op: "xor".to_string(),
                a: new_left,
                b: new_right,
            });
            opt_gate_to_new.insert(cone.inputs.len() + local_idx, new_idx);
        }

        for (i, out_idx) in output_xors.iter().enumerate() {
            let local_signal = optimized_output_signals[i];
            if local_signal == -1 {
                if let Some(matrix) = &cone.matrix64 {
                    let row = matrix[i];
                    if row.count_ones() == 1 {
                        let bit_pos = row.trailing_zeros() as usize;
                        old_to_new.insert(*out_idx, cone.inputs[bit_pos]);
                    }
                }
            } else if (local_signal as usize) < cone.inputs.len() {
                old_to_new.insert(*out_idx, cone.inputs[local_signal as usize]);
            } else {
                if let Some(mapped) = opt_gate_to_new.get(&(local_signal as usize)) {
                    old_to_new.insert(*out_idx, *mapped);
                }
            }
        }

        let mut new_outputs = Vec::with_capacity(circuit.outputs.len());
        for (idx, inv) in &circuit.outputs {
            let new_idx = old_to_new.get(idx).copied().unwrap_or(*idx);
            new_outputs.push((new_idx, *inv));
        }
        return Circuit {
            input_bits: circuit.input_bits,
            output_bits: circuit.output_bits,
            gates: new_gates,
            outputs: new_outputs,
        };
    }

    let stop_at: HashSet<u32> = and_gate_indices.iter().copied().collect();
    let mut and_inputs: Vec<u32> = Vec::new();
    for and_idx in &and_gate_indices {
        let gate_idx = *and_idx - circuit.input_bits;
        if let Some(gate) = circuit.gates.get(gate_idx as usize) {
            match gate {
                Gate::Ternary { a, b, c, .. } => {
                    for v in [*a, *b, *c] {
                        if !stop_at.contains(&v) && v >= circuit.input_bits {
                            and_inputs.push(v);
                        }
                    }
                }
                Gate::Binary { a, b, .. } => {
                    for v in [*a, *b] {
                        if !stop_at.contains(&v) && v >= circuit.input_bits {
                            and_inputs.push(v);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut and_input_xors: Vec<u32> = Vec::new();
    for idx in and_inputs {
        let gate_idx = idx - circuit.input_bits;
        if let Some(Gate::Binary { op, .. }) = circuit.gates.get(gate_idx as usize) {
            if op == "xor" {
                and_input_xors.push(idx);
            }
        }
    }

    let output_indices: Vec<u32> = circuit.outputs.iter().map(|(i, _)| *i).collect();
    let mut post_and_xors: Vec<u32> = Vec::new();
    for idx in output_indices {
        if idx >= circuit.input_bits && !stop_at.contains(&idx) {
            let gate_idx = idx - circuit.input_bits;
            if let Some(Gate::Binary { op, .. }) = circuit.gates.get(gate_idx as usize) {
                if op == "xor" {
                    post_and_xors.push(idx);
                }
            }
        }
    }

    let mut new_gates: Vec<Gate> = Vec::new();
    let mut old_to_new: HashMap<u32, u32> = HashMap::new();
    for i in 0..circuit.input_bits {
        old_to_new.insert(i, i);
    }

    for (g_idx, gate) in circuit.gates.iter().enumerate() {
        let full_idx = circuit.input_bits + g_idx as u32;
        match gate {
            Gate::Ternary { a, b, c, imm8 } => {
                let new_a = old_to_new.get(a).copied().unwrap_or(*a);
                let new_b = old_to_new.get(b).copied().unwrap_or(*b);
                let new_c = old_to_new.get(c).copied().unwrap_or(*c);
                let new_idx = circuit.input_bits + new_gates.len() as u32;
                new_gates.push(Gate::Ternary {
                    a: new_a,
                    b: new_b,
                    c: new_c,
                    imm8: *imm8,
                });
                old_to_new.insert(full_idx, new_idx);
            }
            Gate::Const { value, width } => {
                let new_idx = circuit.input_bits + new_gates.len() as u32;
                new_gates.push(Gate::Const {
                    value: *value,
                    width: *width,
                });
                old_to_new.insert(full_idx, new_idx);
            }
            Gate::Unary { op, a } if op == "not" => {
                let new_a = old_to_new.get(a).copied().unwrap_or(*a);
                let new_idx = circuit.input_bits + new_gates.len() as u32;
                new_gates.push(Gate::Unary {
                    op: "not".to_string(),
                    a: new_a,
                });
                old_to_new.insert(full_idx, new_idx);
            }
            Gate::Binary { op, a, b } if op == "or" => {
                let new_a = old_to_new.get(a).copied().unwrap_or(*a);
                let new_b = old_to_new.get(b).copied().unwrap_or(*b);
                let new_idx = circuit.input_bits + new_gates.len() as u32;
                new_gates.push(Gate::Binary {
                    op: "or".to_string(),
                    a: new_a,
                    b: new_b,
                });
                old_to_new.insert(full_idx, new_idx);
            }
            _ => {}
        }
    }

    if !and_input_xors.is_empty() {
        let pre_cone = LinearCone::from_circuit(circuit, &and_input_xors, &stop_at);
        let upstream_ands: HashSet<u32> =
            pre_cone.inputs.iter().copied().collect::<HashSet<u32>>().intersection(&stop_at).copied().collect();
        if !upstream_ands.is_empty() {
            return circuit.clone();
        }

        let mut updated_pre_inputs: Vec<u32> = Vec::new();
        for inp in &pre_cone.inputs {
            updated_pre_inputs.push(old_to_new.get(inp).copied().unwrap_or(*inp));
        }

        let pre_gates = pre_cone.to_xor_circuit_optimized(64);
        let pre_output_signals = pre_cone.get_output_signals_optimized(64);

        let mut opt_gate_to_new: HashMap<usize, u32> = HashMap::new();
        for (local_idx, (left, right)) in pre_gates.iter().enumerate() {
            let new_left = if *left < updated_pre_inputs.len() {
                updated_pre_inputs[*left]
            } else {
                *opt_gate_to_new.get(left).unwrap()
            };
            let new_right = if *right < updated_pre_inputs.len() {
                updated_pre_inputs[*right]
            } else {
                *opt_gate_to_new.get(right).unwrap()
            };
            let new_idx = circuit.input_bits + new_gates.len() as u32;
            new_gates.push(Gate::Binary {
                op: "xor".to_string(),
                a: new_left,
                b: new_right,
            });
            opt_gate_to_new.insert(updated_pre_inputs.len() + local_idx, new_idx);
        }

        for (i, out_idx) in and_input_xors.iter().enumerate() {
            let local_signal = pre_output_signals[i];
            if local_signal == -1 {
                if let Some(matrix) = &pre_cone.matrix64 {
                    let row = matrix[i];
                    if row.count_ones() == 1 {
                        let bit_pos = row.trailing_zeros() as usize;
                        old_to_new.insert(*out_idx, updated_pre_inputs[bit_pos]);
                    }
                }
            } else if (local_signal as usize) < updated_pre_inputs.len() {
                old_to_new.insert(*out_idx, updated_pre_inputs[local_signal as usize]);
            } else if let Some(mapped) = opt_gate_to_new.get(&(local_signal as usize)) {
                old_to_new.insert(*out_idx, *mapped);
            }
        }
    }

    let mut and_only_sorted = and_only_indices.clone();
    and_only_sorted.sort_unstable();
    for and_idx in and_only_sorted {
        let gate_idx = and_idx - circuit.input_bits;
        if let Some(Gate::Binary { a, b, .. }) = circuit.gates.get(gate_idx as usize) {
            let new_left = old_to_new.get(a).copied().unwrap_or(*a);
            let new_right = old_to_new.get(b).copied().unwrap_or(*b);
            let new_idx = circuit.input_bits + new_gates.len() as u32;
            new_gates.push(Gate::Binary {
                op: "and".to_string(),
                a: new_left,
                b: new_right,
            });
            old_to_new.insert(and_idx, new_idx);
        }
    }

    if !post_and_xors.is_empty() {
        let post_cone = LinearCone::from_circuit(circuit, &post_and_xors, &stop_at);
        let mut updated_inputs: Vec<u32> = Vec::new();
        for inp in &post_cone.inputs {
            updated_inputs.push(old_to_new.get(inp).copied().unwrap_or(*inp));
        }

        let post_gates = post_cone.to_xor_circuit_optimized(64);
        let post_output_signals = post_cone.get_output_signals_optimized(64);

        let mut opt_gate_to_new: HashMap<usize, u32> = HashMap::new();
        for (local_idx, (left, right)) in post_gates.iter().enumerate() {
            let new_left = if *left < updated_inputs.len() {
                updated_inputs[*left]
            } else {
                *opt_gate_to_new.get(left).unwrap()
            };
            let new_right = if *right < updated_inputs.len() {
                updated_inputs[*right]
            } else {
                *opt_gate_to_new.get(right).unwrap()
            };
            let new_idx = circuit.input_bits + new_gates.len() as u32;
            new_gates.push(Gate::Binary {
                op: "xor".to_string(),
                a: new_left,
                b: new_right,
            });
            opt_gate_to_new.insert(updated_inputs.len() + local_idx, new_idx);
        }

        for (i, out_idx) in post_and_xors.iter().enumerate() {
            let local_signal = post_output_signals[i];
            if local_signal == -1 {
                if let Some(matrix) = &post_cone.matrix64 {
                    let row = matrix[i];
                    if row.count_ones() == 1 {
                        let bit_pos = row.trailing_zeros() as usize;
                        old_to_new.insert(*out_idx, updated_inputs[bit_pos]);
                    }
                }
            } else if (local_signal as usize) < updated_inputs.len() {
                old_to_new.insert(*out_idx, updated_inputs[local_signal as usize]);
            } else if let Some(mapped) = opt_gate_to_new.get(&(local_signal as usize)) {
                old_to_new.insert(*out_idx, *mapped);
            }
        }
    }

    let mut new_outputs: Vec<(u32, bool)> = Vec::with_capacity(circuit.outputs.len());
    for (idx, inv) in &circuit.outputs {
        let new_idx = old_to_new.get(idx).copied().unwrap_or(*idx);
        new_outputs.push((new_idx, *inv));
    }

    Circuit {
        input_bits: circuit.input_bits,
        output_bits: circuit.output_bits,
        gates: new_gates,
        outputs: new_outputs,
    }
}

#[derive(Clone, Debug)]
struct GateInfo {
    op: String,
    inputs: Vec<u32>,
    imm8: Option<u8>,
    fanout: u32,
}

fn build_gate_info(gates: &[Gate], input_bits: u32) -> Vec<GateInfo> {
    let mut infos: Vec<GateInfo> = Vec::with_capacity(gates.len());
    let mut fanout_counts = vec![0u32; input_bits as usize + gates.len()];
    for gate in gates {
        let (op, inputs, imm8) = match gate {
            Gate::Const { .. } => ("const".to_string(), Vec::new(), None),
            Gate::Unary { op, a } => (op.clone(), vec![*a], None),
            Gate::Binary { op, a, b } => (op.clone(), vec![*a, *b], None),
            Gate::Ternary { a, b, c, imm8 } => {
                ("ternary".to_string(), vec![*a, *b, *c], Some(*imm8))
            }
        };
        for inp in &inputs {
            if (*inp as usize) < fanout_counts.len() {
                fanout_counts[*inp as usize] += 1;
            }
        }
        infos.push(GateInfo {
            op,
            inputs,
            imm8,
            fanout: 0,
        });
    }
    for (i, info) in infos.iter_mut().enumerate() {
        let idx = input_bits as usize + i;
        if idx < fanout_counts.len() {
            info.fanout = fanout_counts[idx];
        }
    }
    infos
}

fn compute_imm8(op1: &str, op2: &str, negate_result: bool) -> u8 {
    let mut imm8 = 0u8;
    for i in 0..8 {
        let mut a = (i >> 2) & 1;
        let mut b = (i >> 1) & 1;
        let mut c = i & 1;
        let inner = eval_op(op2, a, b);
        let mut result = eval_op(op1, inner, c);
        if negate_result {
            result ^= 1;
        }
        if result == 1 {
            imm8 |= 1 << i;
        }
    }
    imm8
}

fn compute_imm8_swapped(op1: &str, op2: &str) -> u8 {
    let mut imm8 = 0u8;
    for i in 0..8 {
        let mut a = (i >> 2) & 1;
        let mut b = (i >> 1) & 1;
        let mut c = i & 1;
        let inner = eval_op(op2, a, b);
        let result = eval_op(op1, c, inner);
        if result == 1 {
            imm8 |= 1 << i;
        }
    }
    imm8
}

fn eval_op(op: &str, x: u8, y: u8) -> u8 {
    match op {
        "and" => x & y,
        "or" => x | y,
        "xor" => x ^ y,
        _ => 0,
    }
}

fn find_ternary_patterns(
    gate_infos: &[GateInfo],
    input_bits: u32,
) -> Vec<(usize, usize, u32, u32, u32, u8)> {
    let mut patterns = Vec::new();
    for (outer_idx, outer_info) in gate_infos.iter().enumerate() {
        if outer_info.op != "and" && outer_info.op != "or" && outer_info.op != "xor" {
            continue;
        }
        if outer_info.inputs.len() != 2 {
            continue;
        }
        let outer_in0 = outer_info.inputs[0];
        let outer_in1 = outer_info.inputs[1];
        let candidates = [(outer_in0, outer_in1), (outer_in1, outer_in0)];
        for (inner_idx_node, other_idx) in candidates {
            if inner_idx_node < input_bits {
                continue;
            }
            let inner_gate_idx = (inner_idx_node - input_bits) as usize;
            if inner_gate_idx >= gate_infos.len() {
                continue;
            }
            let inner_info = &gate_infos[inner_gate_idx];
            if inner_info.op != "and" && inner_info.op != "or" && inner_info.op != "xor" {
                continue;
            }
            if inner_info.fanout != 1 {
                continue;
            }
            if inner_info.inputs.len() != 2 {
                continue;
            }
            let (inner_in0, inner_in1) = (inner_info.inputs[0], inner_info.inputs[1]);
            let imm8 = if inner_idx_node == outer_in0 {
                compute_imm8(&outer_info.op, &inner_info.op, false)
            } else {
                compute_imm8_swapped(&outer_info.op, &inner_info.op)
            };
            patterns.push((
                outer_idx,
                inner_gate_idx,
                inner_in0,
                inner_in1,
                other_idx,
                imm8,
            ));
        }
    }
    patterns
}

fn remap_gate_inputs(gate: &Gate, idx_remap: &HashMap<u32, u32>, input_bits: u32) -> Gate {
    match gate {
        Gate::Const { value, width } => Gate::Const {
            value: *value,
            width: *width,
        },
        Gate::Unary { op, a } => {
            let new_a = if *a >= input_bits {
                idx_remap.get(a).copied().unwrap_or(*a)
            } else {
                *a
            };
            Gate::Unary {
                op: op.clone(),
                a: new_a,
            }
        }
        Gate::Binary { op, a, b } => {
            let new_a = if *a >= input_bits {
                idx_remap.get(a).copied().unwrap_or(*a)
            } else {
                *a
            };
            let new_b = if *b >= input_bits {
                idx_remap.get(b).copied().unwrap_or(*b)
            } else {
                *b
            };
            Gate::Binary {
                op: op.clone(),
                a: new_a,
                b: new_b,
            }
        }
        Gate::Ternary { a, b, c, imm8 } => {
            let new_a = if *a >= input_bits {
                idx_remap.get(a).copied().unwrap_or(*a)
            } else {
                *a
            };
            let new_b = if *b >= input_bits {
                idx_remap.get(b).copied().unwrap_or(*b)
            } else {
                *b
            };
            let new_c = if *c >= input_bits {
                idx_remap.get(c).copied().unwrap_or(*c)
            } else {
                *c
            };
            Gate::Ternary {
                a: new_a,
                b: new_b,
                c: new_c,
                imm8: *imm8,
            }
        }
    }
}

fn apply_ternary_synthesis(
    gates: Vec<Gate>,
    input_bits: u32,
    outputs: &[(u32, bool)],
) -> (Vec<Gate>, Vec<(u32, bool)>) {
    let gate_infos = build_gate_info(&gates, input_bits);
    let patterns = find_ternary_patterns(&gate_infos, input_bits);
    if patterns.is_empty() {
        return (gates, outputs.to_vec());
    }
    let mut used_gates: HashSet<usize> = HashSet::new();
    let mut selected: Vec<(usize, usize, u32, u32, u32, u8)> = Vec::new();
    for pattern in patterns {
        let (outer_idx, inner_idx, a, b, c, imm8) = pattern;
        if used_gates.contains(&outer_idx) || used_gates.contains(&inner_idx) {
            continue;
        }
        selected.push((outer_idx, inner_idx, a, b, c, imm8));
        used_gates.insert(outer_idx);
        used_gates.insert(inner_idx);
    }
    if selected.is_empty() {
        return (gates, outputs.to_vec());
    }
    let mut merged_gates: HashSet<usize> = HashSet::new();
    let mut replacement_map: HashMap<usize, (u32, u32, u32, u8)> = HashMap::new();
    for (outer_idx, inner_idx, a, b, c, imm8) in &selected {
        merged_gates.insert(*inner_idx);
        replacement_map.insert(*outer_idx, (*a, *b, *c, *imm8));
    }
    let mut new_gates: Vec<Gate> = Vec::new();
    let mut idx_remap: HashMap<u32, u32> = HashMap::new();
    for (i, gate) in gates.iter().enumerate() {
        if merged_gates.contains(&i) {
            continue;
        }
        let old_node = input_bits + i as u32;
        let new_idx = input_bits + new_gates.len() as u32;
        idx_remap.insert(old_node, new_idx);
        if let Some((a, b, c, imm8)) = replacement_map.get(&i) {
            let a_new = if *a >= input_bits {
                idx_remap.get(a).copied().unwrap_or(*a)
            } else {
                *a
            };
            let b_new = if *b >= input_bits {
                idx_remap.get(b).copied().unwrap_or(*b)
            } else {
                *b
            };
            let c_new = if *c >= input_bits {
                idx_remap.get(c).copied().unwrap_or(*c)
            } else {
                *c
            };
            new_gates.push(Gate::Ternary {
                a: a_new,
                b: b_new,
                c: c_new,
                imm8: *imm8,
            });
        } else {
            let remapped = remap_gate_inputs(gate, &idx_remap, input_bits);
            new_gates.push(remapped);
        }
    }
    for (outer_idx, inner_idx, _, _, _, _) in &selected {
        let inner_node = input_bits + *inner_idx as u32;
        let outer_node = input_bits + *outer_idx as u32;
        if let Some(mapped) = idx_remap.get(&outer_node).copied() {
            idx_remap.insert(inner_node, mapped);
        }
    }
    let mut new_outputs = Vec::with_capacity(outputs.len());
    for (out_idx, inv) in outputs {
        let new_idx = if *out_idx >= input_bits {
            idx_remap.get(out_idx).copied().unwrap_or(*out_idx)
        } else {
            *out_idx
        };
        new_outputs.push((new_idx, *inv));
    }
    (new_gates, new_outputs)
}

fn find_andnot_patterns(
    gate_infos: &[GateInfo],
    input_bits: u32,
) -> Vec<(usize, usize, u32, u32)> {
    let mut patterns = Vec::new();
    for (and_idx, info) in gate_infos.iter().enumerate() {
        if info.op != "and" {
            continue;
        }
        if info.inputs.len() != 2 {
            continue;
        }
        let in0 = info.inputs[0];
        let in1 = info.inputs[1];
        for (not_candidate, other) in [(in0, in1), (in1, in0)] {
            if not_candidate < input_bits {
                continue;
            }
            let not_gate_idx = (not_candidate - input_bits) as usize;
            if not_gate_idx >= gate_infos.len() {
                continue;
            }
            let not_info = &gate_infos[not_gate_idx];
            if not_info.op != "not" || not_info.fanout != 1 {
                continue;
            }
            if not_info.inputs.is_empty() {
                continue;
            }
            let x = not_info.inputs[0];
            patterns.push((and_idx, not_gate_idx, x, other));
        }
    }
    patterns
}

fn apply_andnot_optimization(
    mut gates: Vec<Gate>,
    input_bits: u32,
    outputs: &[(u32, bool)],
) -> (Vec<Gate>, Vec<(u32, bool)>) {
    let mut idx_remap: HashMap<u32, u32> = HashMap::new();
    loop {
        let gate_infos = build_gate_info(&gates, input_bits);
        let patterns = find_andnot_patterns(&gate_infos, input_bits);
        if patterns.is_empty() {
            break;
        }
        let (and_idx, not_idx, x, y) = patterns[0];
        let mut new_gates: Vec<Gate> = Vec::new();
        let mut pass_remap: HashMap<u32, u32> = HashMap::new();
        for (i, gate) in gates.iter().enumerate() {
            if i == not_idx {
                continue;
            } else if i == and_idx {
                let new_idx = input_bits + new_gates.len() as u32;
                let x_new = if x >= input_bits {
                    pass_remap.get(&x).copied().unwrap_or(x)
                } else {
                    x
                };
                let y_new = if y >= input_bits {
                    pass_remap.get(&y).copied().unwrap_or(y)
                } else {
                    y
                };
                new_gates.push(Gate::Binary {
                    op: "andn".to_string(),
                    a: x_new,
                    b: y_new,
                });
                pass_remap.insert(input_bits + i as u32, new_idx);
            } else {
                let new_idx = input_bits + new_gates.len() as u32;
                let remapped = remap_gate_inputs(gate, &pass_remap, input_bits);
                new_gates.push(remapped);
                pass_remap.insert(input_bits + i as u32, new_idx);
            }
        }
        for (old_idx, new_idx) in pass_remap.iter() {
            idx_remap.insert(*old_idx, *new_idx);
        }
        gates = new_gates;
    }
    let mut new_outputs = Vec::with_capacity(outputs.len());
    for (out_idx, inv) in outputs {
        let new_idx = if *out_idx >= input_bits {
            idx_remap.get(out_idx).copied().unwrap_or(*out_idx)
        } else {
            *out_idx
        };
        new_outputs.push((new_idx, *inv));
    }
    (gates, new_outputs)
}

fn eliminate_double_nots(
    mut gates: Vec<Gate>,
    input_bits: u32,
    outputs: &[(u32, bool)],
) -> (Vec<Gate>, Vec<(u32, bool)>) {
    let mut idx_remap: HashMap<u32, u32> = HashMap::new();
    loop {
        let gate_infos = build_gate_info(&gates, input_bits);
        let mut found: Option<(usize, usize, u32)> = None;
        for (i, info) in gate_infos.iter().enumerate() {
            if info.op != "not" || info.inputs.is_empty() {
                continue;
            }
            let inp = info.inputs[0];
            if inp < input_bits {
                continue;
            }
            let inner_idx = (inp - input_bits) as usize;
            if inner_idx >= gate_infos.len() {
                continue;
            }
            let inner_info = &gate_infos[inner_idx];
            if inner_info.op != "not" || inner_info.fanout != 1 {
                continue;
            }
            let x = inner_info.inputs[0];
            found = Some((i, inner_idx, x));
            break;
        }
        if found.is_none() {
            break;
        }
        let (outer_idx, inner_idx, x) = found.unwrap();
        let mut new_gates: Vec<Gate> = Vec::new();
        let mut pass_remap: HashMap<u32, u32> = HashMap::new();
        for (i, gate) in gates.iter().enumerate() {
            if i == inner_idx || i == outer_idx {
                continue;
            }
            let new_idx = input_bits + new_gates.len() as u32;
            let remapped = remap_gate_inputs(gate, &pass_remap, input_bits);
            new_gates.push(remapped);
            pass_remap.insert(input_bits + i as u32, new_idx);
        }
        pass_remap.insert(input_bits + outer_idx as u32, x);
        for (old_idx, new_idx) in pass_remap.iter() {
            idx_remap.insert(*old_idx, *new_idx);
        }
        gates = new_gates;
    }
    let mut new_outputs = Vec::with_capacity(outputs.len());
    for (out_idx, inv) in outputs {
        let new_idx = if *out_idx >= input_bits {
            idx_remap.get(out_idx).copied().unwrap_or(*out_idx)
        } else {
            *out_idx
        };
        new_outputs.push((new_idx, *inv));
    }
    (gates, new_outputs)
}

struct BinReader<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> BinReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, off: 0 }
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.off >= self.data.len() {
            bail!("unexpected eof");
        }
        let v = self.data[self.off];
        self.off += 1;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32> {
        if self.off + 4 > self.data.len() {
            bail!("unexpected eof");
        }
        let v = u32::from_le_bytes(
            self.data[self.off..self.off + 4]
                .try_into()
                .unwrap(),
        );
        self.off += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.off + 8 > self.data.len() {
            bail!("unexpected eof");
        }
        let v = u64::from_le_bytes(
            self.data[self.off..self.off + 8]
                .try_into()
                .unwrap(),
        );
        self.off += 8;
        Ok(v)
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let n = self.read_u32()? as usize;
        if self.off + n > self.data.len() {
            bail!("unexpected eof");
        }
        let b = self.data[self.off..self.off + n].to_vec();
        self.off += n;
        Ok(b)
    }

    fn read_big_uint(&mut self) -> Result<BigUint> {
        let n = self.read_u32()? as usize;
        if n == 0 {
            return Ok(BigUint::zero());
        }
        if self.off + n > self.data.len() {
            bail!("unexpected eof");
        }
        let b = &self.data[self.off..self.off + n];
        self.off += n;
        Ok(BigUint::from_bytes_le(b))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.format.as_str() != "bin" {
        bail!("tick_lower_rs supports only --format bin");
    }
    let timing = std::env::var("STC_TIMING")
        .ok()
        .map(|v| v != "0" && v != "false" && v != "no")
        .unwrap_or(false);
    let t0 = std::time::Instant::now();
    let data = fs::read(&args.input)
        .with_context(|| format!("read {}", args.input.display()))?;
    if timing {
        eprintln!(
            "[timing] tick_lower_rs:read_input: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    let t0 = std::time::Instant::now();
    let (strings, inputs, outputs, state, nodes, reset_map, next_map, out_map) =
        read_tick_ir_bin_nodes(&data)?;
    if timing {
        eprintln!(
            "[timing] tick_lower_rs:parse_input: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    let t0 = std::time::Instant::now();
    let ctx_types = build_types(&inputs, &state)?;
    let (circuit, layout) = lower_nodes_to_circuit(
        &strings,
        &ctx_types,
        &inputs,
        &outputs,
        &state,
        &nodes,
        &reset_map,
        &next_map,
        &out_map,
    )?;
    if timing {
        eprintln!(
            "[timing] tick_lower_rs:lower: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    let t0 = std::time::Instant::now();
    if args.output_format.as_str() == "bin" {
        let gates = parse_gates(&circuit.gates)?;
        let c = Circuit {
            input_bits: circuit.input_bits,
            output_bits: circuit.output_bits,
            gates,
            outputs: circuit.outputs.clone(),
        };
        write_circuit_bin(&c, &args.output)?;
    } else {
        let out = serde_json::to_vec_pretty(&circuit)?;
        fs::write(&args.output, out)?;
    }
    if timing {
        eprintln!(
            "[timing] tick_lower_rs:write_output: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    let layout_path = layout_output_path(&args.output, args.output_format.as_str());
    if args.output_format.as_str() == "bin" {
        write_layout_bin(&layout, &layout_path)?;
    } else {
        let layout_json = layout_to_json(&layout);
        let layout_bytes = serde_json::to_vec_pretty(&layout_json)?;
        fs::write(&layout_path, layout_bytes)?;
    }
    if timing {
        eprintln!(
            "[timing] tick_lower_rs:write_layout: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn layout_output_path(output: &PathBuf, format: &str) -> PathBuf {
    let mut p = output.clone();
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("circuit_state");
    let suffix = if format == "bin" { "bin" } else { "json" };
    let layout_name = format!("{stem}_layout.{suffix}");
    p.set_file_name(layout_name);
    p
}

fn build_types(inputs: &HashMap<String, TypeVal>, state: &HashMap<String, TypeVal>) -> Result<HashMap<String, Ty>> {
    let mut types = HashMap::new();
    for (k, v) in inputs {
        types.insert(k.clone(), typeval_to_ty(v));
    }
    for (k, v) in state {
        types.insert(k.clone(), typeval_to_ty(v));
    }
    Ok(types)
}

fn typeval_to_ty(tv: &TypeVal) -> Ty {
    match tv {
        TypeVal::Bool => Ty::Bool,
        TypeVal::BitVec { width } => Ty::BitVec { width: *width as u64 },
        TypeVal::Float { width } => Ty::Float { width: *width as u64 },
        TypeVal::Simd { lane_width, lanes } => Ty::Simd {
            lane_width: *lane_width as u64,
            lanes: *lanes as u64,
        },
    }
}

fn width_bits(t: &Ty) -> Result<u32> {
    match t {
        Ty::Bool => Ok(1),
        Ty::BitVec { width } => Ok(*width as u32),
        _ => bail!("unsupported type for bit-level lowering"),
    }
}

struct LowerCtx<'a> {
    nodes: &'a [Node],
    strings: &'a [String],
    ctx_types: &'a HashMap<String, Ty>,
    in_offsets: &'a HashMap<String, u32>,
    st_offsets: &'a HashMap<String, u32>,
    input_bits: u32,
    gates: Vec<Value>,
    const_cache: HashMap<(u32, u32), u32>,
    memo_bits: HashMap<u32, Vec<u32>>,
    memo_width: HashMap<u32, u32>,
}

impl<'a> LowerCtx<'a> {
    fn new(
        nodes: &'a [Node],
        strings: &'a [String],
        ctx_types: &'a HashMap<String, Ty>,
        in_offsets: &'a HashMap<String, u32>,
        st_offsets: &'a HashMap<String, u32>,
        input_bits: u32,
    ) -> Self {
        Self {
            nodes,
            strings,
            ctx_types,
            in_offsets,
            st_offsets,
            input_bits,
            gates: Vec::new(),
            const_cache: HashMap::new(),
            memo_bits: HashMap::new(),
            memo_width: HashMap::new(),
        }
    }

    fn gate_const(&mut self, value: u32, width: u32) -> u32 {
        let key = (value, width);
        if let Some(existing) = self.const_cache.get(&key) {
            return *existing;
        }
        let idx = self.input_bits + self.gates.len() as u32;
        self.gates.push(Value::Array(vec![
            Value::String("const".to_string()),
            Value::from(value),
            Value::from(width),
        ]));
        self.const_cache.insert(key, idx);
        idx
    }

    fn gate_bin(&mut self, op: &str, a: u32, b: u32) -> u32 {
        let idx = self.input_bits + self.gates.len() as u32;
        self.gates.push(Value::Array(vec![
            Value::String(op.to_string()),
            Value::from(a),
            Value::from(b),
        ]));
        idx
    }

    fn gate_ternary(&mut self, a: u32, b: u32, c: u32, imm8: u8) -> u32 {
        let idx = self.input_bits + self.gates.len() as u32;
        self.gates.push(Value::Array(vec![
            Value::String("ternary".to_string()),
            Value::from(a),
            Value::from(b),
            Value::from(c),
            Value::from(imm8 as u64),
        ]));
        idx
    }

    fn node_width(&mut self, node_id: u32) -> Result<u32> {
        if let Some(w) = self.memo_width.get(&node_id) {
            return Ok(*w);
        }
        let node = self
            .nodes
            .get(node_id as usize)
            .ok_or_else(|| anyhow::anyhow!("node id out of range"))?;
        let w = match node {
            Node::Var { name_idx } => {
                let name = self
                    .strings
                    .get(*name_idx as usize)
                    .ok_or_else(|| anyhow::anyhow!("bad name idx"))?;
                width_bits(
                    self.ctx_types
                        .get(name)
                        .ok_or_else(|| anyhow::anyhow!("unknown var {}", name))?,
                )?
            }
            Node::BoolConst(_) => 1,
            Node::BitVecConst { width, .. } => *width,
            Node::FloatConst { .. } => bail!("float not supported in bit-level lowering"),
            Node::SimdConst { .. } => bail!("simd not supported in bit-level lowering"),
            Node::Unary { x, .. } => self.node_width(*x)?,
            Node::Binary { op, a, .. } => match op {
                100 | 101 | 102 | 103 | 104 => 1,
                _ => self.node_width(*a)?,
            },
            Node::Ternary { a, .. } => self.node_width(*a)?,
            Node::Mux { a, .. } => self.node_width(*a)?,
            Node::Concat { parts } => {
                let mut sum = 0u32;
                for p in parts {
                    sum += self.node_width(*p)?;
                }
                sum
            }
            Node::Slice { width, .. } => *width,
            Node::Lut8 { .. } => 8,
            Node::TernaryLut { a, .. } => self.node_width(*a)?,
            Node::Bitcast { to, .. } => width_bits(&typeval_to_ty(to))?,
            Node::Delay { x, .. } => self.node_width(*x)?,
            _ => bail!("unsupported expression for bit-level lowering"),
        };
        self.memo_width.insert(node_id, w);
        Ok(w)
    }

    fn lower_bits(&mut self, node_id: u32) -> Result<Vec<u32>> {
        if let Some(cached) = self.memo_bits.get(&node_id) {
            return Ok(cached.clone());
        }
        let node = self
            .nodes
            .get(node_id as usize)
            .ok_or_else(|| anyhow::anyhow!("node id out of range"))?
            .clone();
        let w = self.node_width(node_id)?;
        let bits = match node {
            Node::Var { name_idx } => {
                let name = self
                    .strings
                    .get(name_idx as usize)
                    .ok_or_else(|| anyhow::anyhow!("bad name idx"))?;
                let mut out = Vec::with_capacity(w as usize);
                if let Some(off) = self.in_offsets.get(name) {
                    for i in 0..w {
                        out.push(off + i);
                    }
                } else if let Some(off) = self.st_offsets.get(name) {
                    for i in 0..w {
                        out.push(off + i);
                    }
                } else {
                    bail!("unknown packed var {}", name);
                }
                out
            }
            Node::BoolConst(v) => vec![self.gate_const(if v { 1 } else { 0 }, 1)],
            Node::BitVecConst { value, .. } => {
                let mut out = Vec::with_capacity(w as usize);
                for i in 0..w {
                    let bit = if (&value >> i) & BigUint::from(1u8) == BigUint::from(1u8) {
                        1
                    } else {
                        0
                    };
                    out.push(self.gate_const(bit, 1));
                }
                out
            }
            Node::Slice { x, offset, width } => {
                let xb = self.lower_bits(x)?;
                let start = offset as usize;
                let end = (offset + width) as usize;
                if end > xb.len() {
                    let xnode = self
                        .nodes
                        .get(x as usize)
                        .cloned()
                        .unwrap_or(Node::BoolConst(false));
                    bail!(
                        "slice out of range: node={} {:?} x={} {:?} offset={} width={} x_bits={}",
                        node_id,
                        node,
                        x,
                        xnode,
                        offset,
                        width,
                        xb.len()
                    );
                }
                xb[start..end].to_vec()
            }
            Node::Concat { parts } => {
                let mut out = Vec::with_capacity(w as usize);
                for p in parts.into_iter().rev() {
                    out.extend(self.lower_bits(p)?);
                }
                if out.len() != w as usize {
                    bail!("concat width mismatch");
                }
                out
            }
            Node::Unary { op, x } => {
                if op != 0 {
                    bail!("unsupported unary op");
                }
                let xb = self.lower_bits(x)?;
                xb.iter().map(|b| self.gate_bin("not", *b, 0)).collect()
            }
            Node::Binary { op, a, b } => match op {
                1 | 2 | 3 => {
                    let ab = self.lower_bits(a)?;
                    let bb = self.lower_bits(b)?;
                    if ab.len() != bb.len() {
                        bail!("binary op width mismatch");
                    }
                    let op_name = match op {
                        1 => "and",
                        2 => "or",
                        _ => "xor",
                    };
                    ab.iter()
                        .zip(bb.iter())
                        .map(|(x, y)| self.gate_bin(op_name, *x, *y))
                        .collect()
                }
                4 => {
                    let ab = self.lower_bits(a)?;
                    let bb = self.lower_bits(b)?;
                    self.lower_add_bits(&ab, &bb)?
                }
                5 => {
                    let ab = self.lower_bits(a)?;
                    let bb = self.lower_bits(b)?;
                    self.lower_sub_bits(&ab, &bb)?
                }
                6 | 7 | 8 => {
                    let sh = self.shift_amount_const(b)?;
                    let xb = self.lower_bits(a)?;
                    let sh = sh as usize;
                    let w = xb.len();
                    let mut out = Vec::with_capacity(w);
                    if op == 6 {
                        for _ in 0..sh {
                            out.push(self.gate_const(0, 1));
                        }
                        out.extend_from_slice(&xb[..w.saturating_sub(sh)]);
                    } else if op == 7 {
                        out.extend_from_slice(&xb[sh.min(w)..]);
                        for _ in 0..sh {
                            out.push(self.gate_const(0, 1));
                        }
                    } else {
                        let sign = if w > 0 { xb[w - 1] } else { self.gate_const(0, 1) };
                        out.extend_from_slice(&xb[sh.min(w)..]);
                        for _ in 0..sh {
                            out.push(sign);
                        }
                    }
                    out.truncate(w);
                    out
                }
                100 => self.lower_eq(a, b)?,
                101 => self.lower_ult(a, b)?,
                102 => {
                    let lt = self.lower_ult(a, b)?[0];
                    let eq = self.lower_eq(a, b)?[0];
                    vec![self.gate_bin("or", lt, eq)]
                }
                103 => {
                    let le = {
                        let lt = self.lower_ult(a, b)?[0];
                        let eq = self.lower_eq(a, b)?[0];
                        self.gate_bin("or", lt, eq)
                    };
                    vec![self.gate_bin("not", le, 0)]
                }
                104 => {
                    let lt = self.lower_ult(a, b)?[0];
                    vec![self.gate_bin("not", lt, 0)]
                }
                _ => bail!("unsupported binary op"),
            },
            Node::TernaryLut { a, b, c, imm8 } => {
                let ab = self.lower_bits(a)?;
                let bb = self.lower_bits(b)?;
                let cb = self.lower_bits(c)?;
                if ab.len() != bb.len() || ab.len() != cb.len() {
                    bail!("ternary width mismatch");
                }
                ab.iter()
                    .zip(bb.iter())
                    .zip(cb.iter())
                    .map(|((x, y), z)| self.gate_ternary(*x, *y, *z, imm8))
                    .collect()
            }
            Node::Mux { cond, a, b } => {
                let c = self.lower_bits(cond)?;
                if c.len() != 1 {
                    bail!("mux condition not bool");
                }
                let cbit = c[0];
                let tb = self.lower_bits(a)?;
                let fb = self.lower_bits(b)?;
                if tb.len() != fb.len() {
                    bail!("mux width mismatch");
                }
                let c_not = self.gate_bin("not", cbit, 0);
                tb.iter()
                    .zip(fb.iter())
                    .map(|(t, f)| {
                        let ct = self.gate_bin("and", cbit, *t);
                        let cnf = self.gate_bin("and", c_not, *f);
                        self.gate_bin("or", ct, cnf)
                    })
                    .collect()
            }
            Node::Delay { x, .. } => self.lower_bits(x)?,
            Node::Bitcast { x, .. } => self.lower_bits(x)?,
            Node::Lut8 { x, table } => self.lower_lut8(x, &table)?,
            _ => bail!("unsupported expression for bit-level lowering"),
        };
        self.memo_bits.insert(node_id, bits.clone());
        Ok(bits)
    }

    fn shift_amount_const(&mut self, node_id: u32) -> Result<u32> {
        let node = self
            .nodes
            .get(node_id as usize)
            .ok_or_else(|| anyhow::anyhow!("node id out of range"))?;
        if let Node::BitVecConst { value, .. } = node {
            Ok(value.to_u32_digits().get(0).cloned().unwrap_or(0))
        } else {
            bail!("shift requires constant shift amount");
        }
    }

    fn lower_eq(&mut self, a: u32, b: u32) -> Result<Vec<u32>> {
        let ab = self.lower_bits(a)?;
        let bb = self.lower_bits(b)?;
        if ab.len() != bb.len() {
            bail!("eq width mismatch");
        }
        if ab.len() == 1 {
            let x = self.gate_bin("xor", ab[0], bb[0]);
            return Ok(vec![self.gate_bin("not", x, 0)]);
        }
        let mut acc: Option<u32> = None;
        for (abit, bbit) in ab.iter().zip(bb.iter()) {
            let x = self.gate_bin("xor", *abit, *bbit);
            let xnor = self.gate_bin("not", x, 0);
            acc = Some(match acc {
                None => xnor,
                Some(prev) => self.gate_bin("and", prev, xnor),
            });
        }
        Ok(vec![acc.unwrap()])
    }

    fn lower_ult(&mut self, a: u32, b: u32) -> Result<Vec<u32>> {
        let ab = self.lower_bits(a)?;
        let bb = self.lower_bits(b)?;
        if ab.len() != bb.len() {
            bail!("ult width mismatch");
        }
        if ab.is_empty() {
            bail!("ult on empty bitvector");
        }
        let mut eq = self.gate_const(1, 1);
        let mut lt = self.gate_const(0, 1);
        for i in (0..ab.len()).rev() {
            let abit = ab[i];
            let bbit = bb[i];
            let not_a = self.gate_bin("not", abit, 0);
            let a_lt_b = self.gate_bin("and", not_a, bbit);
            let lt_here = self.gate_bin("and", eq, a_lt_b);
            lt = self.gate_bin("or", lt, lt_here);
            let axb = self.gate_bin("xor", abit, bbit);
            let xnor = self.gate_bin("not", axb, 0);
            eq = self.gate_bin("and", eq, xnor);
        }
        Ok(vec![lt])
    }

    fn lower_add_bits(&mut self, ab: &[u32], bb: &[u32]) -> Result<Vec<u32>> {
        if ab.len() != bb.len() {
            bail!("add width mismatch");
        }
        if ab.is_empty() {
            return Ok(vec![]);
        }
        let mut result = Vec::with_capacity(ab.len());
        let mut carry = self.gate_const(0, 1);
        for (a, b) in ab.iter().zip(bb.iter()) {
            let s_partial = self.gate_bin("xor", *a, *b);
            let s = self.gate_bin("xor", s_partial, carry);
            result.push(s);
            let ab_and = self.gate_bin("and", *a, *b);
            let c_and = self.gate_bin("and", carry, s_partial);
            carry = self.gate_bin("or", ab_and, c_and);
        }
        Ok(result)
    }

    fn lower_sub_bits(&mut self, ab: &[u32], bb: &[u32]) -> Result<Vec<u32>> {
        if ab.len() != bb.len() {
            bail!("sub width mismatch");
        }
        if ab.is_empty() {
            return Ok(vec![]);
        }
        let mut result = Vec::with_capacity(ab.len());
        let mut carry = self.gate_const(1, 1);
        for (a, b) in ab.iter().zip(bb.iter()) {
            let b_not = self.gate_bin("not", *b, 0);
            let s_partial = self.gate_bin("xor", *a, b_not);
            let s = self.gate_bin("xor", s_partial, carry);
            result.push(s);
            let ab_and = self.gate_bin("and", *a, b_not);
            let c_and = self.gate_bin("and", carry, s_partial);
            carry = self.gate_bin("or", ab_and, c_and);
        }
        Ok(result)
    }

    fn lower_lut8(&mut self, x: u32, table: &[u8]) -> Result<Vec<u32>> {
        if table.len() != 256 {
            bail!("lut8 table length");
        }
        let xb = self.lower_bits(x)?;
        if xb.len() != 8 {
            bail!("lut8 input width");
        }
        let mut outputs: Vec<u32> = Vec::with_capacity(8);
        for out_bit in 0..8u32 {
            let mut acc: Option<u32> = None;
            for idx in 0..256u32 {
                if (table[idx as usize] & (1 << out_bit)) == 0 {
                    continue;
                }
                let mut term: Option<u32> = None;
                for bit in 0..8 {
                    let bit_val = (idx >> bit) & 1;
                    let v = xb[bit as usize];
                    let lit = if bit_val == 1 {
                        v
                    } else {
                        self.gate_bin("not", v, 0)
                    };
                    term = Some(match term {
                        None => lit,
                        Some(prev) => self.gate_bin("and", prev, lit),
                    });
                }
                if let Some(term_val) = term {
                    acc = Some(match acc {
                        None => term_val,
                        Some(prev) => self.gate_bin("or", prev, term_val),
                    });
                }
            }
            outputs.push(acc.unwrap_or_else(|| self.gate_const(0, 1)));
        }
        Ok(outputs)
    }
}

fn lower_nodes_to_circuit(
    strings: &[String],
    ctx_types: &HashMap<String, Ty>,
    inputs: &HashMap<String, TypeVal>,
    outputs: &HashMap<String, TypeVal>,
    state: &HashMap<String, TypeVal>,
    nodes: &[Node],
    reset_map: &HashMap<String, u32>,
    next_map: &HashMap<String, u32>,
    out_map: &HashMap<String, u32>,
) -> Result<(CircuitStateOut, PackedLayout)> {
    let mut input_order: Vec<String> = inputs.keys().cloned().collect();
    let mut state_order: Vec<String> = state.keys().cloned().collect();
    let mut output_order: Vec<String> = outputs.keys().cloned().collect();
    input_order.sort();
    state_order.sort();
    output_order.sort();

    let mut in_offsets = HashMap::new();
    let mut st_offsets = HashMap::new();
    let mut out_offsets = HashMap::new();
    let mut nx_offsets = HashMap::new();

    let mut off: u32 = 0;
    for name in &input_order {
        let t = inputs.get(name).unwrap();
        let w = width_bits(&typeval_to_ty(t))?;
        in_offsets.insert(name.clone(), off);
        off += w;
    }
    for name in &state_order {
        let t = state.get(name).unwrap();
        let w = width_bits(&typeval_to_ty(t))?;
        st_offsets.insert(name.clone(), off);
        off += w;
    }
    let input_bits = off;

    let mut out_off: u32 = 0;
    for name in &output_order {
        let t = outputs.get(name).unwrap();
        let w = width_bits(&typeval_to_ty(t))?;
        out_offsets.insert(name.clone(), out_off);
        out_off += w;
    }
    for name in &state_order {
        let t = state.get(name).unwrap();
        let w = width_bits(&typeval_to_ty(t))?;
        nx_offsets.insert(name.clone(), out_off);
        out_off += w;
    }
    let output_bits = out_off;

    let layout = build_layout(
        &input_order,
        &state_order,
        &output_order,
        &in_offsets,
        &st_offsets,
        &out_offsets,
        &nx_offsets,
        input_bits,
        output_bits,
        inputs,
        outputs,
        state,
    );

    let has_reset = inputs
        .get("reset")
        .map(|t| matches!(t, TypeVal::Bool))
        .unwrap_or(false);
    let reset_bit = in_offsets.get("reset").copied();

    let mut ctx = LowerCtx::new(nodes, strings, ctx_types, &in_offsets, &st_offsets, input_bits);

    let mut packed_outputs: Vec<u32> = Vec::new();
    for name in &output_order {
        let node_id = *out_map
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing output expr for {}", name))?;
        let bits = ctx.lower_bits(node_id)?;
        packed_outputs.extend(bits);
    }
    for name in &state_order {
        let next_id = *next_map
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing next_state expr for {}", name))?;
        let next_bits = ctx.lower_bits(next_id)?;
        if has_reset {
            let reset_bit = reset_bit.ok_or_else(|| anyhow::anyhow!("reset bit missing"))?;
            let reset_expr = reset_map.get(name);
            let reset_bits_expr = if let Some(reset_node) = reset_expr {
                ctx.lower_bits(*reset_node)?
            } else {
                vec![ctx.gate_const(0, 1)]
            };
            if reset_bits_expr.len() != next_bits.len() {
                bail!("reset width mismatch");
            }
            let mut muxed = Vec::with_capacity(next_bits.len());
            let c_not = ctx.gate_bin("not", reset_bit, 0);
            for (a, b) in reset_bits_expr.iter().zip(next_bits.iter()) {
                let ct = ctx.gate_bin("and", reset_bit, *a);
                let cnf = ctx.gate_bin("and", c_not, *b);
                muxed.push(ctx.gate_bin("or", ct, cnf));
            }
            packed_outputs.extend(muxed);
        } else {
            packed_outputs.extend(next_bits);
        }
    }

    if packed_outputs.len() != output_bits as usize {
        bail!("packed output width mismatch");
    }

    let outputs_vec: Vec<(u32, bool)> = packed_outputs
        .iter()
        .map(|idx| (*idx, false))
        .collect();
    let gate_count = ctx.gates.len();
    let mut circuit = CircuitStateOut {
        input_bits,
        output_bits,
        gates: ctx.gates,
        outputs: outputs_vec,
        gate_count,
    };
    if is_truthy_env("STC_SKIP_LINEAR_OPT") {
        return Ok((circuit, layout));
    }
    let want_linear = is_truthy_env("STC_RUST_LINEAR_OPT") || is_truthy_env("STC_FORCE_LINEAR_OPT");
    if want_linear {
        let max_gates: usize = std::env::var("STC_LINEAR_OPT_MAX_GATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50000);
        let force_linear = is_truthy_env("STC_FORCE_LINEAR_OPT");
        if force_linear || circuit.gate_count <= max_gates {
            let gates = parse_gates(&circuit.gates)?;
            let c = Circuit {
                input_bits: circuit.input_bits,
                output_bits: circuit.output_bits,
                gates,
                outputs: circuit.outputs.clone(),
            };
            let optimized = optimize_linear_layers(&c);
            if optimized.gates.len() < c.gates.len() {
                circuit.gates = gates_to_values(&optimized.gates);
                circuit.outputs = optimized.outputs;
                circuit.gate_count = optimized.gates.len();
            }
        }
    }
    if !is_truthy_env("STC_SKIP_GATE_TERNARY")
        && (is_truthy_env("STC_RUST_GATE_TERNARY") || is_truthy_env("STC_FORCE_GATE_TERNARY"))
    {
        let mut gates = parse_gates(&circuit.gates)?;
        let outputs = circuit.outputs.clone();
        let (g1, o1) = eliminate_double_nots(gates, circuit.input_bits, &outputs);
        let (g2, o2) = apply_andnot_optimization(g1, circuit.input_bits, &o1);
        let (g3, o3) = apply_ternary_synthesis(g2, circuit.input_bits, &o2);
        circuit.gates = gates_to_values(&g3);
        circuit.outputs = o3;
        circuit.gate_count = g3.len();
    }
    Ok((circuit, layout))
}

fn build_layout(
    input_order: &[String],
    state_order: &[String],
    output_order: &[String],
    in_offsets: &HashMap<String, u32>,
    st_offsets: &HashMap<String, u32>,
    out_offsets: &HashMap<String, u32>,
    nx_offsets: &HashMap<String, u32>,
    input_bits: u32,
    output_bits: u32,
    inputs: &HashMap<String, TypeVal>,
    outputs: &HashMap<String, TypeVal>,
    state: &HashMap<String, TypeVal>,
) -> PackedLayout {
    let mut inputs_map = HashMap::new();
    for name in input_order {
        let w = width_bits(&typeval_to_ty(inputs.get(name).unwrap())).unwrap_or(0);
        inputs_map.insert(
            name.clone(),
            LayoutEntry {
                lsb: in_offsets[name],
                width: w,
            },
        );
    }
    let mut state_map = HashMap::new();
    for name in state_order {
        let w = width_bits(&typeval_to_ty(state.get(name).unwrap())).unwrap_or(0);
        state_map.insert(
            name.clone(),
            LayoutEntry {
                lsb: st_offsets[name],
                width: w,
            },
        );
    }
    let mut outputs_map = HashMap::new();
    for name in output_order {
        let w = width_bits(&typeval_to_ty(outputs.get(name).unwrap())).unwrap_or(0);
        outputs_map.insert(
            name.clone(),
            LayoutEntry {
                lsb: out_offsets[name],
                width: w,
            },
        );
    }
    let mut next_map = HashMap::new();
    for name in state_order {
        let w = width_bits(&typeval_to_ty(state.get(name).unwrap())).unwrap_or(0);
        next_map.insert(
            name.clone(),
            LayoutEntry {
                lsb: nx_offsets[name],
                width: w,
            },
        );
    }
    PackedLayout {
        inputs: inputs_map,
        state: state_map,
        outputs: outputs_map,
        next_state: next_map,
        input_bits,
        output_bits,
    }
}

fn layout_to_json(layout: &PackedLayout) -> Value {
    let mut inputs_map = serde_json::Map::new();
    for (k, v) in layout.inputs.iter() {
        inputs_map.insert(k.clone(), json!({ "lsb": v.lsb, "width": v.width }));
    }
    let mut state_map = serde_json::Map::new();
    for (k, v) in layout.state.iter() {
        state_map.insert(k.clone(), json!({ "lsb": v.lsb, "width": v.width }));
    }
    let mut outputs_map = serde_json::Map::new();
    for (k, v) in layout.outputs.iter() {
        outputs_map.insert(k.clone(), json!({ "lsb": v.lsb, "width": v.width }));
    }
    let mut next_map = serde_json::Map::new();
    for (k, v) in layout.next_state.iter() {
        next_map.insert(k.clone(), json!({ "lsb": v.lsb, "width": v.width }));
    }
    json!({
        "inputs": Value::Object(inputs_map),
        "state": Value::Object(state_map),
        "outputs": Value::Object(outputs_map),
        "next_state": Value::Object(next_map),
        "input_bits": layout.input_bits,
        "output_bits": layout.output_bits,
    })
}

fn write_circuit_bin(circuit: &Circuit, path: &PathBuf) -> Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"CSB1");
    buf.push(1);
    buf.extend_from_slice(&circuit.input_bits.to_le_bytes());
    buf.extend_from_slice(&circuit.output_bits.to_le_bytes());
    buf.extend_from_slice(&(circuit.gates.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(circuit.outputs.len() as u32).to_le_bytes());
    for gate in circuit.gates.iter() {
        match gate {
            Gate::Ternary { a, b, c, imm8 } => {
                buf.push(5);
                buf.extend_from_slice(&a.to_le_bytes());
                buf.extend_from_slice(&b.to_le_bytes());
                buf.extend_from_slice(&c.to_le_bytes());
                buf.push(*imm8);
            }
            Gate::Const { value, width } => {
                buf.push(4);
                buf.extend_from_slice(&value.to_le_bytes());
                buf.extend_from_slice(&width.to_le_bytes());
            }
            Gate::Unary { op, a } => {
                if op == "not" {
                    buf.push(3);
                    buf.extend_from_slice(&a.to_le_bytes());
                } else {
                    bail!("unsupported unary op for bin: {op}");
                }
            }
            Gate::Binary { op, a, b } => {
                let opcode = match op.as_str() {
                    "xor" => 0,
                    "and" => 1,
                    "or" => 2,
                    "andnot" => 6,
                    "ornot" => 7,
                    _ => bail!("unsupported binary op for bin: {op}"),
                };
                buf.push(opcode);
                buf.extend_from_slice(&a.to_le_bytes());
                buf.extend_from_slice(&b.to_le_bytes());
            }
        }
    }
    for (idx, inv) in circuit.outputs.iter() {
        buf.extend_from_slice(&idx.to_le_bytes());
        buf.push(if *inv { 1 } else { 0 });
    }
    fs::write(path, buf)?;
    Ok(())
}

fn write_layout_bin(layout: &PackedLayout, path: &PathBuf) -> Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"PLB1");
    buf.push(1);
    let mut names: Vec<String> = layout
        .inputs
        .keys()
        .chain(layout.state.keys())
        .chain(layout.outputs.keys())
        .chain(layout.next_state.keys())
        .cloned()
        .collect();
    names.sort();
    names.dedup();
    buf.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for name in names.iter() {
        let bytes = name.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }
    let idx_of = |name: &str| -> Result<u32> {
        names
            .iter()
            .position(|n| n == name)
            .map(|i| i as u32)
            .ok_or_else(|| anyhow::anyhow!("missing name in table"))
    };
    buf.extend_from_slice(&layout.input_bits.to_le_bytes());
    buf.extend_from_slice(&layout.output_bits.to_le_bytes());
    let maps = [
        &layout.inputs,
        &layout.state,
        &layout.outputs,
        &layout.next_state,
    ];
    for m in maps.iter() {
        buf.extend_from_slice(&(m.len() as u32).to_le_bytes());
        for (k, v) in m.iter() {
            let idx = idx_of(k)?;
            buf.extend_from_slice(&idx.to_le_bytes());
            buf.extend_from_slice(&v.lsb.to_le_bytes());
            buf.extend_from_slice(&v.width.to_le_bytes());
        }
    }
    fs::write(path, buf)?;
    Ok(())
}

fn read_tick_ir_bin_nodes(
    data: &[u8],
) -> Result<(
    Vec<String>,
    HashMap<String, TypeVal>,
    HashMap<String, TypeVal>,
    HashMap<String, TypeVal>,
    Vec<Node>,
    HashMap<String, u32>,
    HashMap<String, u32>,
    HashMap<String, u32>,
)> {
    let mut r = BinReader::new(data);
    let version = r.read_u8()?;
    if version != 1 {
        bail!("unsupported bin version {version}");
    }
    let strings_len = r.read_u32()? as usize;
    let mut strings = Vec::with_capacity(strings_len);
    for _ in 0..strings_len {
        let bytes = r.read_bytes()?;
        strings.push(String::from_utf8(bytes)?);
    }
    let node_len = r.read_u32()? as usize;
    let mut nodes = Vec::with_capacity(node_len);
    for _ in 0..node_len {
        nodes.push(read_node(&mut r)?);
    }
    let inputs = read_type_map(&mut r, &strings)?;
    let outputs = read_type_map(&mut r, &strings)?;
    let state = read_type_map(&mut r, &strings)?;
    let reset_map = read_expr_map(&mut r, &strings)?;
    let next_map = read_expr_map(&mut r, &strings)?;
    let out_map = read_expr_map(&mut r, &strings)?;
    Ok((strings, inputs, outputs, state, nodes, reset_map, next_map, out_map))
}

fn read_type_map(
    r: &mut BinReader,
    strings: &[String],
) -> Result<HashMap<String, TypeVal>> {
    let len = r.read_u32()? as usize;
    let mut map = HashMap::new();
    for _ in 0..len {
        let idx = r.read_u32()? as usize;
        let name = strings.get(idx).ok_or_else(|| anyhow::anyhow!("bad string idx"))?;
        let tv = read_type_val(r)?;
        map.insert(name.clone(), tv);
    }
    Ok(map)
}

fn read_expr_map(
    r: &mut BinReader,
    strings: &[String],
) -> Result<HashMap<String, u32>> {
    let len = r.read_u32()? as usize;
    let mut map = HashMap::new();
    for _ in 0..len {
        let idx = r.read_u32()? as usize;
        let name = strings.get(idx).ok_or_else(|| anyhow::anyhow!("bad string idx"))?;
        let node_id = r.read_u32()?;
        map.insert(name.clone(), node_id);
    }
    Ok(map)
}

fn read_type_val(r: &mut BinReader) -> Result<TypeVal> {
    let tag = r.read_u8()?;
    Ok(match tag {
        0 => TypeVal::Bool,
        1 => TypeVal::BitVec {
            width: r.read_u32()?,
        },
        2 => TypeVal::Float {
            width: r.read_u32()?,
        },
        3 => TypeVal::Simd {
            lane_width: r.read_u32()?,
            lanes: r.read_u32()?,
        },
        _ => bail!("bad type tag"),
    })
}

fn read_node(r: &mut BinReader) -> Result<Node> {
    let opcode = r.read_u8()?;
    Ok(match opcode {
        0 => Node::Var {
            name_idx: r.read_u32()?,
        },
        1 => Node::BoolConst(r.read_u8()? != 0),
        2 => Node::BitVecConst {
            width: r.read_u32()?,
            value: r.read_big_uint()?,
        },
        3 => Node::FloatConst {
            width: r.read_u32()?,
            bits: r.read_u64()?,
        },
        4 => Node::SimdConst {
            lane_width: r.read_u32()?,
            lanes: r.read_u32()?,
            value: r.read_big_uint()?,
        },
        5 => Node::Unary {
            op: 0,
            x: r.read_u32()?,
        },
        6 => Node::Binary {
            op: 1,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        7 => Node::Binary {
            op: 2,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        8 => Node::Binary {
            op: 3,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        9 => Node::Binary {
            op: 4,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        10 => Node::Binary {
            op: 5,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        11 => Node::Binary {
            op: 6,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        12 => Node::Binary {
            op: 7,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        13 => Node::Binary {
            op: 8,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        60 => Node::Binary {
            op: 100,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        61 => Node::Binary {
            op: 101,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        62 => Node::Binary {
            op: 102,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        63 => Node::Binary {
            op: 103,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        64 => Node::Binary {
            op: 104,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        65 => Node::Mux {
            cond: r.read_u32()?,
            a: r.read_u32()?,
            b: r.read_u32()?,
        },
        66 => Node::Concat {
            parts: read_u32_vec(r)?,
        },
        67 => Node::Slice {
            x: r.read_u32()?,
            offset: r.read_u32()?,
            width: r.read_u32()?,
        },
        68 => Node::Lut8 {
            x: r.read_u32()?,
            table: read_u8_vec(r)?,
        },
        69 => Node::TernaryLut {
            a: r.read_u32()?,
            b: r.read_u32()?,
            c: r.read_u32()?,
            imm8: r.read_u8()?,
        },
        70 => Node::Bitcast {
            to: read_type_val(r)?,
            x: r.read_u32()?,
        },
        73 => Node::Delay {
            x: r.read_u32()?,
            ticks: r.read_u32()?,
        },
        _ => bail!("unsupported opcode {opcode}"),
    })
}

fn read_u32_vec(r: &mut BinReader) -> Result<Vec<u32>> {
    let n = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.read_u32()?);
    }
    Ok(out)
}

fn read_u8_vec(r: &mut BinReader) -> Result<Vec<u8>> {
    let n = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.read_u8()?);
    }
    Ok(out)
}
