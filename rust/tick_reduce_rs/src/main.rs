use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use num_bigint::BigUint;
use num_traits::Zero;
use serde_json::Value;

#[derive(Debug, Clone)]
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
    #[arg(long, default_value = "json")]
    format: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let timing = std::env::var("STC_TIMING")
        .ok()
        .map(|v| v != "0" && v != "false" && v != "no")
        .unwrap_or(false);
    let t0 = std::time::Instant::now();
    let data = fs::read(&args.input)
        .with_context(|| format!("read {}", args.input.display()))?;
    if timing {
        eprintln!(
            "[timing] rust_reduce_rs:read_input: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
    match args.format.as_str() {
        "bin" => {
            let t0 = std::time::Instant::now();
            let (strings, inputs, outputs, state, nodes, reset_map, next_map, out_map) =
                read_tick_ir_bin_nodes(&data)?;
            if timing {
                eprintln!(
                    "[timing] rust_reduce_rs:parse_input: {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            let t0 = std::time::Instant::now();
            let (reduced_nodes, new_reset, new_next, new_out) =
                reduce_nodes(&nodes, &reset_map, &next_map, &out_map);
            if timing {
                eprintln!(
                    "[timing] rust_reduce_rs:reduce: {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            let t0 = std::time::Instant::now();
            let out = write_tick_ir_bin_nodes(
                &strings,
                &inputs,
                &outputs,
                &state,
                reduced_nodes,
                &new_reset,
                &new_next,
                &new_out,
            )?;
            if timing {
                eprintln!(
                    "[timing] rust_reduce_rs:serialize_output: {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            fs::write(&args.output, out)?;
        }
        "json" | "msgpack" => {
            let t0 = std::time::Instant::now();
            let ir: Value = match args.format.as_str() {
                "json" => serde_json::from_slice(&data)?,
                "msgpack" => rmp_serde::from_slice(&data)?,
                _ => unreachable!(),
            };
            if timing {
                eprintln!(
                    "[timing] rust_reduce_rs:parse_input: {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            let t0 = std::time::Instant::now();
            let types = build_types(&ir)?;
            let reduced = reduce_tick_ir(&ir, &types)?;
            if timing {
                eprintln!(
                    "[timing] rust_reduce_rs:reduce: {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            let t0 = std::time::Instant::now();
            let out = match args.format.as_str() {
                "json" => serde_json::to_vec_pretty(&reduced)?,
                "msgpack" => rmp_serde::to_vec(&reduced)?,
                _ => unreachable!(),
            };
            if timing {
                eprintln!(
                    "[timing] rust_reduce_rs:serialize_output: {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            fs::write(&args.output, out)?;
        }
        other => bail!("unknown format {other}"),
    }
    Ok(())
}

fn build_types(ir: &Value) -> Result<HashMap<String, Ty>> {
    let mut types = HashMap::new();
    for section in ["inputs", "state"] {
        let map = ir
            .get(section)
            .and_then(|v| v.as_object())
            .with_context(|| format!("missing {section}"))?;
        for (k, v) in map {
            let t = parse_type(v)?;
            types.insert(k.clone(), t);
        }
    }
    Ok(types)
}

fn parse_type(v: &Value) -> Result<Ty> {
    let kind = v
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("type missing kind"))?;
    match kind {
        "bool" => Ok(Ty::Bool),
        "bitvec" => Ok(Ty::BitVec {
            width: v.get("width").and_then(|v| v.as_u64()).unwrap_or(0),
        }),
        "float" => Ok(Ty::Float {
            width: v.get("width").and_then(|v| v.as_u64()).unwrap_or(0),
        }),
        "simd" => Ok(Ty::Simd {
            lane_width: v.get("lane_width").and_then(|v| v.as_u64()).unwrap_or(0),
            lanes: v.get("lanes").and_then(|v| v.as_u64()).unwrap_or(0),
        }),
        _ => bail!("unknown type kind {kind}"),
    }
}

fn reduce_tick_ir(ir: &Value, _types: &HashMap<String, Ty>) -> Result<Value> {
    Ok(ir.clone())
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
        let v = u32::from_le_bytes(self.data[self.off..self.off + 4].try_into()?);
        self.off += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.off + 8 > self.data.len() {
            bail!("unexpected eof");
        }
        let v = u64::from_le_bytes(self.data[self.off..self.off + 8].try_into()?);
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
            return Ok(BigUint::from(0u8));
        }
        if self.off + n > self.data.len() {
            bail!("unexpected eof");
        }
        let v = BigUint::from_bytes_le(&self.data[self.off..self.off + n]);
        self.off += n;
        Ok(v)
    }
}

fn read_tick_ir_bin(data: &[u8]) -> Result<Value> {
    let mut r = BinReader::new(data);
    let version = r.read_u8()?;
    if version != 1 {
        bail!("unsupported tick ir bin version");
    }
    let str_count = r.read_u32()? as usize;
    let mut strings = Vec::with_capacity(str_count);
    for _ in 0..str_count {
        strings.push(String::from_utf8(r.read_bytes()?)?);
    }
    let nodes = read_nodes(&mut r, &strings)?;
    let inputs = read_type_map(&mut r, &strings)?;
    let outputs = read_type_map(&mut r, &strings)?;
    let state = read_type_map(&mut r, &strings)?;
    let reset_state = read_expr_map(&mut r, &strings, &nodes)?;
    let next_state = read_expr_map(&mut r, &strings, &nodes)?;
    let output_exprs = read_expr_map(&mut r, &strings, &nodes)?;
    let mut out = serde_json::Map::new();
    out.insert("schema_version".to_string(), Value::from(1));
    out.insert("name".to_string(), Value::from("tickir"));
    out.insert("inputs".to_string(), Value::Object(inputs));
    out.insert("outputs".to_string(), Value::Object(outputs));
    out.insert("state".to_string(), Value::Object(state));
    out.insert("reset_state".to_string(), Value::Object(reset_state));
    out.insert("next_state".to_string(), Value::Object(next_state));
    out.insert("output_exprs".to_string(), Value::Object(output_exprs));
    Ok(Value::Object(out))
}

fn read_tick_ir_bin_nodes(
    data: &[u8],
) -> Result<(
    Vec<String>,
    Vec<(u32, TypeVal)>,
    Vec<(u32, TypeVal)>,
    Vec<(u32, TypeVal)>,
    Vec<Node>,
    Vec<(u32, u32)>,
    Vec<(u32, u32)>,
    Vec<(u32, u32)>,
)> {
    let mut r = BinReader::new(data);
    let version = r.read_u8()?;
    if version != 1 {
        bail!("unsupported tick ir bin version");
    }
    let str_count = r.read_u32()? as usize;
    let mut strings = Vec::with_capacity(str_count);
    for _ in 0..str_count {
        strings.push(String::from_utf8(r.read_bytes()?)?);
    }
    let nodes = read_nodes_bin(&mut r)?;
    let inputs = read_type_map_bin(&mut r)?;
    let outputs = read_type_map_bin(&mut r)?;
    let state = read_type_map_bin(&mut r)?;
    let reset_state = read_expr_map_bin(&mut r)?;
    let next_state = read_expr_map_bin(&mut r)?;
    let output_exprs = read_expr_map_bin(&mut r)?;
    Ok((
        strings,
        inputs,
        outputs,
        state,
        nodes,
        reset_state,
        next_state,
        output_exprs,
    ))
}

fn read_type_map_bin(r: &mut BinReader<'_>) -> Result<Vec<(u32, TypeVal)>> {
    let count = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let key = r.read_u32()?;
        let ty = read_type_val(r)?;
        out.push((key, ty));
    }
    Ok(out)
}

fn read_expr_map_bin(r: &mut BinReader<'_>) -> Result<Vec<(u32, u32)>> {
    let count = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let key = r.read_u32()?;
        let idx = r.read_u32()?;
        out.push((key, idx));
    }
    Ok(out)
}

fn read_type_val(r: &mut BinReader<'_>) -> Result<TypeVal> {
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

fn read_nodes_bin(r: &mut BinReader<'_>) -> Result<Vec<Node>> {
    let node_count = r.read_u32()? as usize;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let opcode = r.read_u8()?;
        let kind = KIND_LIST
            .get(opcode as usize)
            .ok_or_else(|| anyhow::anyhow!("bad opcode"))?;
        let node = match *kind {
            "var" => Node::Var {
                name_idx: r.read_u32()?,
            },
            "bool_const" => Node::BoolConst(r.read_u8()? != 0),
            "bitvec_const" => Node::BitVecConst {
                width: r.read_u32()?,
                value: r.read_big_uint()?,
            },
            "float_const" => Node::FloatConst {
                width: r.read_u32()?,
                bits: r.read_u64()?,
            },
            "simd_const" => Node::SimdConst {
                lane_width: r.read_u32()?,
                lanes: r.read_u32()?,
                value: r.read_big_uint()?,
            },
            "not" | "simd_not" | "fneg" | "fabs" | "fsqrt" | "simd_fneg" | "simd_fabs"
            | "simd_fsqrt" => Node::Unary {
                op: opcode,
                x: r.read_u32()?,
            },
            "and"
            | "or"
            | "xor"
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "eq"
            | "ult"
            | "ule"
            | "ugt"
            | "uge"
            | "simd_add"
            | "simd_sub"
            | "simd_add_sat_u"
            | "simd_sub_sat_u"
            | "simd_add_sat_s"
            | "simd_sub_sat_s"
            | "simd_mul_lo"
            | "simd_mul_hi_u"
            | "simd_mul_hi_s"
            | "simd_madd_s16"
            | "simd_unpack_lo"
            | "simd_unpack_hi"
            | "simd_pack_ss16_to_8"
            | "simd_pack_us16_to_8"
            | "simd_pack_ss32_to_16"
            | "simd_min_u"
            | "simd_max_u"
            | "simd_min_s"
            | "simd_max_s"
            | "simd_eq"
            | "simd_and"
            | "simd_or"
            | "simd_xor"
            | "simd_ult"
            | "simd_ule"
            | "simd_ugt"
            | "simd_uge"
            | "simd_slt"
            | "simd_sle"
            | "simd_sgt"
            | "simd_sge"
            | "fadd"
            | "fsub"
            | "fmul"
            | "fdiv"
            | "feq"
            | "flt"
            | "fle"
            | "fne"
            | "simd_fadd"
            | "simd_fsub"
            | "simd_fmul"
            | "simd_fdiv"
            | "simd_fcmp_eq"
            | "simd_fcmp_lt"
            | "simd_fcmp_le"
            | "simd_fcmp_ne"
            | "shl"
            | "lshr"
            | "ashr"
            | "simd_shl"
            | "simd_lshr"
            | "simd_ashr" => Node::Binary {
                op: opcode,
                a: r.read_u32()?,
                b: r.read_u32()?,
            },
            "mux" => Node::Mux {
                cond: r.read_u32()?,
                a: r.read_u32()?,
                b: r.read_u32()?,
            },
            "concat" => {
                let n = r.read_u32()? as usize;
                let mut parts = Vec::with_capacity(n);
                for _ in 0..n {
                    parts.push(r.read_u32()?);
                }
                Node::Concat { parts }
            }
            "slice" => Node::Slice {
                x: r.read_u32()?,
                offset: r.read_u32()?,
                width: r.read_u32()?,
            },
            "lut8" => {
                let x = r.read_u32()?;
                let n = r.read_u32()? as usize;
                let mut table = Vec::with_capacity(n);
                for _ in 0..n {
                    table.push(r.read_u8()?);
                }
                Node::Lut8 { x, table }
            }
            "ternary_lut" => Node::TernaryLut {
                a: r.read_u32()?,
                b: r.read_u32()?,
                c: r.read_u32()?,
                imm8: r.read_u8()?,
            },
            "bitcast" => Node::Bitcast {
                to: read_type_val(r)?,
                x: r.read_u32()?,
            },
            "bit_transpose" => Node::BitTranspose {
                x: r.read_u32()?,
                lane_width: r.read_u32()?,
                lanes: r.read_u32()?,
            },
            "rotl" | "rotr" => Node::Rot {
                op: opcode,
                x: r.read_u32()?,
                sh: r.read_u32()?,
            },
            "delay" => Node::Delay {
                x: r.read_u32()?,
                ticks: r.read_u32()?,
            },
            "simd_add_masked" | "simd_sub_masked" | "simd_blend" => Node::Masked {
                op: opcode,
                mask: r.read_u32()?,
                a: r.read_u32()?,
                b: r.read_u32()?,
            },
            "simd_mask_expand" => Node::MaskExpand {
                to: read_type_val(r)?,
                x: r.read_u32()?,
            },
            "simd_mask_pack" => Node::MaskPack { x: r.read_u32()? },
            "simd_zext_lo" | "simd_sext_lo" => Node::Ext {
                op: opcode,
                to: read_type_val(r)?,
                x: r.read_u32()?,
            },
            "simd_splat" => Node::Splat {
                to: read_type_val(r)?,
                x: r.read_u32()?,
            },
            "simd_extract_lane" => Node::ExtractLane {
                x: r.read_u32()?,
                lane: r.read_u32()?,
            },
            "simd_insert_lane" => Node::InsertLane {
                x: r.read_u32()?,
                lane: r.read_u32()?,
                value: r.read_u32()?,
            },
            "simd_shuffle" => {
                let x = r.read_u32()?;
                let n = r.read_u32()? as usize;
                let mut indices = Vec::with_capacity(n);
                for _ in 0..n {
                    indices.push(r.read_u32()?);
                }
                Node::Shuffle { x, indices }
            }
            "simd_ffma" => Node::Ternary {
                op: opcode,
                a: r.read_u32()?,
                b: r.read_u32()?,
                c: r.read_u32()?,
            },
            _ => bail!("unsupported kind {kind}"),
        };
        nodes.push(node);
    }
    Ok(nodes)
}

fn read_type_map(
    r: &mut BinReader<'_>,
    strings: &[String],
) -> Result<serde_json::Map<String, Value>> {
    let mut out = serde_json::Map::new();
    let count = r.read_u32()? as usize;
    for _ in 0..count {
        let key = strings[r.read_u32()? as usize].clone();
        let ty = read_type_value(r)?;
        out.insert(key, ty);
    }
    Ok(out)
}

fn read_expr_map(
    r: &mut BinReader<'_>,
    strings: &[String],
    nodes: &[Value],
) -> Result<serde_json::Map<String, Value>> {
    let mut out = serde_json::Map::new();
    let count = r.read_u32()? as usize;
    for _ in 0..count {
        let key = strings[r.read_u32()? as usize].clone();
        let idx = r.read_u32()? as usize;
        out.insert(key, nodes[idx].clone());
    }
    Ok(out)
}

fn read_type_value(r: &mut BinReader<'_>) -> Result<Value> {
    let tag = r.read_u8()?;
    let mut obj = serde_json::Map::new();
    match tag {
        0 => {
            obj.insert("kind".to_string(), Value::from("bool"));
        }
        1 => {
            obj.insert("kind".to_string(), Value::from("bitvec"));
            obj.insert("width".to_string(), Value::from(r.read_u32()?));
        }
        2 => {
            obj.insert("kind".to_string(), Value::from("float"));
            obj.insert("width".to_string(), Value::from(r.read_u32()?));
        }
        3 => {
            obj.insert("kind".to_string(), Value::from("simd"));
            obj.insert("lane_width".to_string(), Value::from(r.read_u32()?));
            obj.insert("lanes".to_string(), Value::from(r.read_u32()?));
        }
        _ => bail!("bad type tag"),
    }
    Ok(Value::Object(obj))
}

fn read_nodes(r: &mut BinReader<'_>, strings: &[String]) -> Result<Vec<Value>> {
    let node_count = r.read_u32()? as usize;
    let mut nodes: Vec<Value> = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let opcode = r.read_u8()? as usize;
        let kind = KIND_LIST
            .get(opcode)
            .ok_or_else(|| anyhow::anyhow!("bad opcode"))?;
        let mut obj = serde_json::Map::new();
        obj.insert("kind".to_string(), Value::from(*kind));
        match *kind {
            "var" => {
                obj.insert("name".to_string(), Value::from(strings[r.read_u32()? as usize].clone()));
            }
            "bool_const" => {
                obj.insert("value".to_string(), Value::from(r.read_u8()? != 0));
            }
            "bitvec_const" => {
                obj.insert("width".to_string(), Value::from(r.read_u32()?));
                let v = r.read_big_uint()?;
                obj.insert("value".to_string(), Value::from(v.to_string()));
            }
            "float_const" => {
                obj.insert("width".to_string(), Value::from(r.read_u32()?));
                obj.insert("bits".to_string(), Value::from(r.read_u64()?));
            }
            "simd_const" => {
                obj.insert("lane_width".to_string(), Value::from(r.read_u32()?));
                obj.insert("lanes".to_string(), Value::from(r.read_u32()?));
                let v = r.read_big_uint()?;
                obj.insert("value".to_string(), Value::from(v.to_string()));
            }
            _ => {
                // Fallback: treat as binary nodes with a/b when present.
                if matches!(
                    *kind,
                    "not" | "simd_not" | "fneg" | "fabs" | "fsqrt" | "simd_fneg" | "simd_fabs" | "simd_fsqrt"
                ) {
                    let x = nodes[r.read_u32()? as usize].clone();
                    obj.insert("x".to_string(), x);
                } else if matches!(
                    *kind,
                    "and" | "or" | "xor" | "add" | "sub" | "mul" | "div" | "eq" | "ult" | "ule" | "ugt" | "uge"
                ) {
                    let a = nodes[r.read_u32()? as usize].clone();
                    let b = nodes[r.read_u32()? as usize].clone();
                    obj.insert("a".to_string(), a);
                    obj.insert("b".to_string(), b);
                } else {
                    // Generic: skip operands (no-op)
                }
            }
        }
        nodes.push(Value::Object(obj));
    }
    Ok(nodes)
}

const KIND_LIST: [&str; 97] = [
    "var",
    "bool_const",
    "bitvec_const",
    "float_const",
    "simd_const",
    "not",
    "and",
    "or",
    "xor",
    "add",
    "sub",
    "shl",
    "lshr",
    "ashr",
    "simd_add",
    "simd_sub",
    "simd_add_masked",
    "simd_sub_masked",
    "simd_add_sat_u",
    "simd_sub_sat_u",
    "simd_add_sat_s",
    "simd_sub_sat_s",
    "simd_mul_lo",
    "simd_mul_hi_u",
    "simd_mul_hi_s",
    "simd_madd_s16",
    "simd_unpack_lo",
    "simd_unpack_hi",
    "simd_pack_ss16_to_8",
    "simd_pack_us16_to_8",
    "simd_pack_ss32_to_16",
    "simd_mask_expand",
    "simd_mask_pack",
    "simd_min_u",
    "simd_max_u",
    "simd_min_s",
    "simd_max_s",
    "simd_blend",
    "simd_zext_lo",
    "simd_sext_lo",
    "simd_eq",
    "simd_not",
    "simd_and",
    "simd_or",
    "simd_xor",
    "simd_shl",
    "simd_lshr",
    "simd_ashr",
    "simd_ult",
    "simd_ule",
    "simd_ugt",
    "simd_uge",
    "simd_slt",
    "simd_sle",
    "simd_sgt",
    "simd_sge",
    "simd_splat",
    "simd_extract_lane",
    "simd_insert_lane",
    "simd_shuffle",
    "eq",
    "ult",
    "ule",
    "ugt",
    "uge",
    "mux",
    "concat",
    "slice",
    "lut8",
    "ternary_lut",
    "bitcast",
    "rotl",
    "rotr",
    "delay",
    "fneg",
    "fabs",
    "fadd",
    "fsub",
    "fmul",
    "fdiv",
    "fsqrt",
    "feq",
    "flt",
    "fle",
    "fne",
    "simd_fadd",
    "simd_fsub",
    "simd_fmul",
    "simd_fdiv",
    "simd_ffma",
    "simd_fsqrt",
    "simd_fneg",
    "simd_fabs",
    "simd_fcmp_eq",
    "simd_fcmp_lt",
    "simd_fcmp_le",
    "simd_fcmp_ne",
];

// Existing JSON-based reducer functions remain below.

fn reduce_nodes(
    nodes: &[Node],
    reset_map: &[(u32, u32)],
    next_map: &[(u32, u32)],
    out_map: &[(u32, u32)],
) -> (Vec<Node>, Vec<(u32, u32)>, Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let mut memo: Vec<Option<u32>> = vec![None; nodes.len()];
    let mut new_nodes: Vec<Node> = Vec::new();

    fn mask_big(width: u64) -> BigUint {
        if width == 0 {
            return BigUint::from(0u8);
        }
        (BigUint::from(1u8) << width) - BigUint::from(1u8)
    }

    fn simplify_binary(
        op: u8,
        a: u32,
        b: u32,
        nodes: &[Node],
    ) -> Option<Node> {
        let kind = KIND_LIST[op as usize];
        let na = &nodes[a as usize];
        let nb = &nodes[b as usize];
        if kind == "and" || kind == "or" || kind == "xor" || kind == "add" || kind == "sub" || kind == "mul" || kind == "div" {
            if let (Node::BoolConst(av), Node::BoolConst(bv)) = (na, nb) {
                return Some(match kind {
                    "and" => Node::BoolConst(*av && *bv),
                    "or" => Node::BoolConst(*av || *bv),
                    "xor" => Node::BoolConst(*av ^ *bv),
                    _ => Node::BoolConst(false),
                });
            }
            if let (Node::BitVecConst { width, value: av }, Node::BitVecConst { value: bv, .. }) = (na, nb) {
                let mask = mask_big(*width as u64);
                let res = match kind {
                    "and" => av & bv,
                    "or" => av | bv,
                    "xor" => av ^ bv,
                    "add" => (av + bv) & &mask,
                    "sub" => (av - bv) & &mask,
                    "mul" => (av * bv) & &mask,
                    "div" => {
                        if bv.is_zero() {
                            return None;
                        }
                        (av / bv) & &mask
                    }
                    _ => return None,
                };
                return Some(Node::BitVecConst {
                    width: *width,
                    value: res,
                });
            }
            if let (Node::SimdConst { lane_width, lanes, value: av }, Node::SimdConst { value: bv, .. }) = (na, nb) {
                let total = (*lane_width as u64) * (*lanes as u64);
                let mask = mask_big(total);
                let res = match kind {
                    "and" => av & bv,
                    "or" => av | bv,
                    "xor" => av ^ bv,
                    "add" => (av + bv) & &mask,
                    "sub" => (av - bv) & &mask,
                    "mul" => (av * bv) & &mask,
                    "div" => {
                        if bv.is_zero() {
                            return None;
                        }
                        (av / bv) & &mask
                    }
                    _ => return None,
                };
                return Some(Node::SimdConst {
                    lane_width: *lane_width,
                    lanes: *lanes,
                    value: res,
                });
            }
        }
        if kind == "eq" && a == b {
            return Some(Node::BoolConst(true));
        }
        None
    }

    fn concat_const(parts: &[u32], nodes: &[Node]) -> Option<Node> {
        let mut acc = BigUint::from(0u8);
        let mut total = 0u32;
        for p in parts {
            match &nodes[*p as usize] {
                Node::BitVecConst { width, value } => {
                    acc = (acc << *width) | value;
                    total += *width;
                }
                _ => return None,
            }
        }
        Some(Node::BitVecConst {
            width: total,
            value: acc,
        })
    }

    fn reduce_id(
        id: u32,
        nodes: &[Node],
        memo: &mut [Option<u32>],
        new_nodes: &mut Vec<Node>,
    ) -> u32 {
        if let Some(existing) = memo[id as usize] {
            return existing;
        }
        let n = &nodes[id as usize];
        let new_node = match n {
            Node::Var { .. }
            | Node::BoolConst(_)
            | Node::BitVecConst { .. }
            | Node::FloatConst { .. }
            | Node::SimdConst { .. } => n.clone(),
            Node::Unary { op, x } => {
                let rx = reduce_id(*x, nodes, memo, new_nodes);
                match (&nodes[rx as usize], KIND_LIST[*op as usize]) {
                    (Node::BoolConst(v), "not") => Node::BoolConst(!v),
                    (Node::BitVecConst { width, value }, "not") => {
                        let mask = mask_big(*width as u64);
                        Node::BitVecConst {
                            width: *width,
                            value: mask ^ value,
                        }
                    }
                    (Node::SimdConst { lane_width, lanes, value }, "simd_not") => {
                        let total = (*lane_width as u64) * (*lanes as u64);
                        let mask = mask_big(total);
                        Node::SimdConst {
                            lane_width: *lane_width,
                            lanes: *lanes,
                            value: mask ^ value,
                        }
                    }
                    (Node::Unary { op: inner, x: inner_x }, "not")
                        if KIND_LIST[*inner as usize] == "not" =>
                    {
                        return *inner_x;
                    }
                    _ => Node::Unary { op: *op, x: rx },
                }
            }
            Node::Binary { op, a, b } => {
                let ra = reduce_id(*a, nodes, memo, new_nodes);
                let rb = reduce_id(*b, nodes, memo, new_nodes);
                simplify_binary(*op, ra, rb, nodes).unwrap_or(Node::Binary {
                    op: *op,
                    a: ra,
                    b: rb,
                })
            }
            Node::Ternary { op, a, b, c } => Node::Ternary {
                op: *op,
                a: reduce_id(*a, nodes, memo, new_nodes),
                b: reduce_id(*b, nodes, memo, new_nodes),
                c: reduce_id(*c, nodes, memo, new_nodes),
            },
            Node::Mux { cond, a, b } => {
                let rc = reduce_id(*cond, nodes, memo, new_nodes);
                let ra = reduce_id(*a, nodes, memo, new_nodes);
                let rb = reduce_id(*b, nodes, memo, new_nodes);
                if let Node::BoolConst(v) = &nodes[rc as usize] {
                    return if *v { ra } else { rb };
                }
                if ra == rb {
                    return ra;
                }
                Node::Mux {
                    cond: rc,
                    a: ra,
                    b: rb,
                }
            }
            Node::Concat { parts } => {
                let rparts: Vec<u32> = parts
                    .iter()
                    .map(|p| reduce_id(*p, nodes, memo, new_nodes))
                    .collect();
                concat_const(&rparts, nodes).unwrap_or(Node::Concat { parts: rparts })
            }
            Node::Slice { x, offset, width } => {
                let rx = reduce_id(*x, nodes, memo, new_nodes);
                if let Node::BitVecConst { value, .. } = &nodes[rx as usize] {
                    let mask = mask_big(*width as u64);
                    let shifted = (value >> *offset) & mask;
                    Node::BitVecConst {
                        width: *width,
                        value: shifted,
                    }
                } else {
                    Node::Slice {
                        x: rx,
                        offset: *offset,
                        width: *width,
                    }
                }
            }
            Node::Lut8 { x, table } => {
                let rx = reduce_id(*x, nodes, memo, new_nodes);
                if let Node::BitVecConst { value, .. } = &nodes[rx as usize] {
                    let idx = value.to_u64_digits().get(0).cloned().unwrap_or(0) & 0xFF;
                    let out = table.get(idx as usize).cloned().unwrap_or(0);
                    Node::BitVecConst {
                        width: 8,
                        value: BigUint::from(out),
                    }
                } else {
                    Node::Lut8 {
                        x: rx,
                        table: table.clone(),
                    }
                }
            }
            Node::TernaryLut { a, b, c, imm8 } => Node::TernaryLut {
                a: reduce_id(*a, nodes, memo, new_nodes),
                b: reduce_id(*b, nodes, memo, new_nodes),
                c: reduce_id(*c, nodes, memo, new_nodes),
                imm8: *imm8,
            },
            Node::Bitcast { to, x } => Node::Bitcast {
                to: to.clone(),
                x: reduce_id(*x, nodes, memo, new_nodes),
            },
            Node::BitTranspose { x, lane_width, lanes } => Node::BitTranspose {
                x: reduce_id(*x, nodes, memo, new_nodes),
                lane_width: *lane_width,
                lanes: *lanes,
            },
            Node::Rot { op, x, sh } => Node::Rot {
                op: *op,
                x: reduce_id(*x, nodes, memo, new_nodes),
                sh: reduce_id(*sh, nodes, memo, new_nodes),
            },
            Node::Delay { x, ticks } => Node::Delay {
                x: reduce_id(*x, nodes, memo, new_nodes),
                ticks: *ticks,
            },
            Node::Masked { op, mask, a, b } => Node::Masked {
                op: *op,
                mask: reduce_id(*mask, nodes, memo, new_nodes),
                a: reduce_id(*a, nodes, memo, new_nodes),
                b: reduce_id(*b, nodes, memo, new_nodes),
            },
            Node::MaskExpand { to, x } => Node::MaskExpand {
                to: to.clone(),
                x: reduce_id(*x, nodes, memo, new_nodes),
            },
            Node::MaskPack { x } => Node::MaskPack {
                x: reduce_id(*x, nodes, memo, new_nodes),
            },
            Node::Ext { op, to, x } => Node::Ext {
                op: *op,
                to: to.clone(),
                x: reduce_id(*x, nodes, memo, new_nodes),
            },
            Node::Splat { to, x } => Node::Splat {
                to: to.clone(),
                x: reduce_id(*x, nodes, memo, new_nodes),
            },
            Node::ExtractLane { x, lane } => Node::ExtractLane {
                x: reduce_id(*x, nodes, memo, new_nodes),
                lane: *lane,
            },
            Node::InsertLane { x, lane, value } => Node::InsertLane {
                x: reduce_id(*x, nodes, memo, new_nodes),
                lane: *lane,
                value: reduce_id(*value, nodes, memo, new_nodes),
            },
            Node::Shuffle { x, indices } => Node::Shuffle {
                x: reduce_id(*x, nodes, memo, new_nodes),
                indices: indices.clone(),
            },
        };
        let new_id = new_nodes.len() as u32;
        new_nodes.push(new_node);
        memo[id as usize] = Some(new_id);
        new_id
    }

    let mut new_reset = Vec::with_capacity(reset_map.len());
    for (k, id) in reset_map {
        let rid = reduce_id(*id, nodes, &mut memo, &mut new_nodes);
        new_reset.push((*k, rid));
    }
    let mut new_next = Vec::with_capacity(next_map.len());
    for (k, id) in next_map {
        let rid = reduce_id(*id, nodes, &mut memo, &mut new_nodes);
        new_next.push((*k, rid));
    }
    let mut new_out = Vec::with_capacity(out_map.len());
    for (k, id) in out_map {
        let rid = reduce_id(*id, nodes, &mut memo, &mut new_nodes);
        new_out.push((*k, rid));
    }
    (new_nodes, new_reset, new_next, new_out)
}

struct BinWriter {
    buf: Vec<u8>,
}

impl BinWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bytes(&mut self, b: &[u8]) {
        self.write_u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }

    fn write_big_uint(&mut self, v: &BigUint) {
        if v.is_zero() {
            self.write_u32(0);
            return;
        }
        let b = v.to_bytes_le();
        self.write_u32(b.len() as u32);
        self.buf.extend_from_slice(&b);
    }
}

fn write_tick_ir_bin_nodes(
    strings: &[String],
    inputs: &[(u32, TypeVal)],
    outputs: &[(u32, TypeVal)],
    state: &[(u32, TypeVal)],
    nodes: Vec<Node>,
    reset_map: &[(u32, u32)],
    next_map: &[(u32, u32)],
    out_map: &[(u32, u32)],
) -> Result<Vec<u8>> {
    let mut w = BinWriter::new();
    w.write_u8(1);
    w.write_u32(strings.len() as u32);
    for s in strings {
        w.write_bytes(s.as_bytes());
    }
    w.write_u32(nodes.len() as u32);
    for node in &nodes {
        write_node_bin(&mut w, node)?;
    }
    write_type_map_bin(&mut w, inputs);
    write_type_map_bin(&mut w, outputs);
    write_type_map_bin(&mut w, state);
    write_expr_map_bin(&mut w, reset_map);
    write_expr_map_bin(&mut w, next_map);
    write_expr_map_bin(&mut w, out_map);
    Ok(w.buf)
}

fn write_type_map_bin(w: &mut BinWriter, m: &[(u32, TypeVal)]) {
    w.write_u32(m.len() as u32);
    for (k, t) in m {
        w.write_u32(*k);
        write_type_val_bin(w, t);
    }
}

fn write_expr_map_bin(w: &mut BinWriter, m: &[(u32, u32)]) {
    w.write_u32(m.len() as u32);
    for (k, v) in m {
        w.write_u32(*k);
        w.write_u32(*v);
    }
}

fn write_type_val_bin(w: &mut BinWriter, t: &TypeVal) {
    match t {
        TypeVal::Bool => w.write_u8(0),
        TypeVal::BitVec { width } => {
            w.write_u8(1);
            w.write_u32(*width);
        }
        TypeVal::Float { width } => {
            w.write_u8(2);
            w.write_u32(*width);
        }
        TypeVal::Simd { lane_width, lanes } => {
            w.write_u8(3);
            w.write_u32(*lane_width);
            w.write_u32(*lanes);
        }
    }
}

fn opcode_for_kind(kind: &str) -> u8 {
    for (i, k) in KIND_LIST.iter().enumerate() {
        if *k == kind {
            return i as u8;
        }
    }
    0
}

fn write_node_bin(w: &mut BinWriter, node: &Node) -> Result<()> {
    match node {
        Node::Var { name_idx } => {
            w.write_u8(opcode_for_kind("var"));
            w.write_u32(*name_idx);
        }
        Node::BoolConst(v) => {
            w.write_u8(opcode_for_kind("bool_const"));
            w.write_u8(if *v { 1 } else { 0 });
        }
        Node::BitVecConst { width, value } => {
            w.write_u8(opcode_for_kind("bitvec_const"));
            w.write_u32(*width);
            w.write_big_uint(value);
        }
        Node::FloatConst { width, bits } => {
            w.write_u8(opcode_for_kind("float_const"));
            w.write_u32(*width);
            w.write_u64(*bits);
        }
        Node::SimdConst { lane_width, lanes, value } => {
            w.write_u8(opcode_for_kind("simd_const"));
            w.write_u32(*lane_width);
            w.write_u32(*lanes);
            w.write_big_uint(value);
        }
        Node::Unary { op, x } => {
            w.write_u8(*op);
            w.write_u32(*x);
        }
        Node::Binary { op, a, b } => {
            w.write_u8(*op);
            w.write_u32(*a);
            w.write_u32(*b);
        }
        Node::Ternary { op, a, b, c } => {
            w.write_u8(*op);
            w.write_u32(*a);
            w.write_u32(*b);
            w.write_u32(*c);
        }
        Node::Mux { cond, a, b } => {
            w.write_u8(opcode_for_kind("mux"));
            w.write_u32(*cond);
            w.write_u32(*a);
            w.write_u32(*b);
        }
        Node::Concat { parts } => {
            w.write_u8(opcode_for_kind("concat"));
            w.write_u32(parts.len() as u32);
            for p in parts {
                w.write_u32(*p);
            }
        }
        Node::Slice { x, offset, width } => {
            w.write_u8(opcode_for_kind("slice"));
            w.write_u32(*x);
            w.write_u32(*offset);
            w.write_u32(*width);
        }
        Node::Lut8 { x, table } => {
            w.write_u8(opcode_for_kind("lut8"));
            w.write_u32(*x);
            w.write_u32(table.len() as u32);
            for v in table {
                w.write_u8(*v);
            }
        }
        Node::TernaryLut { a, b, c, imm8 } => {
            w.write_u8(opcode_for_kind("ternary_lut"));
            w.write_u32(*a);
            w.write_u32(*b);
            w.write_u32(*c);
            w.write_u8(*imm8);
        }
        Node::Bitcast { to, x } => {
            w.write_u8(opcode_for_kind("bitcast"));
            write_type_val_bin(w, to);
            w.write_u32(*x);
        }
        Node::BitTranspose { x, lane_width, lanes } => {
            w.write_u8(opcode_for_kind("bit_transpose"));
            w.write_u32(*x);
            w.write_u32(*lane_width);
            w.write_u32(*lanes);
        }
        Node::Rot { op, x, sh } => {
            w.write_u8(*op);
            w.write_u32(*x);
            w.write_u32(*sh);
        }
        Node::Delay { x, ticks } => {
            w.write_u8(opcode_for_kind("delay"));
            w.write_u32(*x);
            w.write_u32(*ticks);
        }
        Node::Masked { op, mask, a, b } => {
            w.write_u8(*op);
            w.write_u32(*mask);
            w.write_u32(*a);
            w.write_u32(*b);
        }
        Node::MaskExpand { to, x } => {
            w.write_u8(opcode_for_kind("simd_mask_expand"));
            write_type_val_bin(w, to);
            w.write_u32(*x);
        }
        Node::MaskPack { x } => {
            w.write_u8(opcode_for_kind("simd_mask_pack"));
            w.write_u32(*x);
        }
        Node::Ext { op, to, x } => {
            w.write_u8(*op);
            write_type_val_bin(w, to);
            w.write_u32(*x);
        }
        Node::Splat { to, x } => {
            w.write_u8(opcode_for_kind("simd_splat"));
            write_type_val_bin(w, to);
            w.write_u32(*x);
        }
        Node::ExtractLane { x, lane } => {
            w.write_u8(opcode_for_kind("simd_extract_lane"));
            w.write_u32(*x);
            w.write_u32(*lane);
        }
        Node::InsertLane { x, lane, value } => {
            w.write_u8(opcode_for_kind("simd_insert_lane"));
            w.write_u32(*x);
            w.write_u32(*lane);
            w.write_u32(*value);
        }
        Node::Shuffle { x, indices } => {
            w.write_u8(opcode_for_kind("simd_shuffle"));
            w.write_u32(*x);
            w.write_u32(indices.len() as u32);
            for i in indices {
                w.write_u32(*i);
            }
        }
    }
    Ok(())
}
