use std::env;
use std::time::Instant;

use aes_cpu_bench::{encrypt_blocks_fixed_key, Block};

fn parse_arg_usize(args: &[String], name: &str, default: usize) -> usize {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == name {
            return args[i + 1].parse::<usize>().unwrap();
        }
        i += 1;
    }
    default
}

fn parse_arg_f64(args: &[String], name: &str, default: f64) -> f64 {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == name {
            return args[i + 1].parse::<f64>().unwrap();
        }
        i += 1;
    }
    default
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let blocks_n = parse_arg_usize(&args, "--blocks", 1 << 18);
    let min_seconds = parse_arg_f64(&args, "--min-seconds", 0.2);

    let mut blocks: Vec<Block> = vec![Block::default(); blocks_n];
    encrypt_blocks_fixed_key(&mut blocks[..1]);

    let mut iters: u64 = 1;
    let (elapsed, iters_done) = loop {
        let start = Instant::now();
        for _ in 0..iters {
            encrypt_blocks_fixed_key(&mut blocks);
        }
        let dt = start.elapsed();
        if dt.as_secs_f64() >= min_seconds || iters > (1 << 30) {
            break (dt, iters);
        }
        iters = iters.saturating_mul(2);
    };

    let secs = elapsed.as_secs_f64();
    let bytes_total = (blocks_n as f64) * 16.0 * (iters_done as f64);
    let mib_s = (bytes_total / secs) / (1024.0 * 1024.0);
    println!(
        "cpu_rust_aes128 blocks={} iters={} time_s={:.6} throughput_mib_s={:.2}",
        blocks_n, iters_done, secs, mib_s
    );
}
