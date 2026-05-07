use std::{env, error::Error, fmt, fs, ops::Index, path::PathBuf};

use json::{JsonValue, number::Number, object::Object};

fn main() {
    println!("cargo:rerun-if-env-changed=BENCH_PARAMETERS");
    println!("cargo:rerun-if-env-changed=QUBITS");
    println!("cargo:rerun-if-env-changed=ARGS");
    let mut rs: Vec<String> = vec![];
    let json_path_str = env::var("BENCH_PARAMETERS").ok();
    if let Some(path) = json_path_str.as_ref() {
        println!("cargo:rerun-if-changed={}",path);
    }
    let json_path = json_path_str.map(PathBuf::from);
    let json_str = json_path
        .map(fs::read_to_string)
        .map(Result::ok)
        .unwrap_or(None);
    let json_data = json_str
        .as_ref()
        .map(String::as_str)
        .map(json::parse)
        .map(Result::ok)
        .unwrap_or(None);

    let json_data = json_data
        .map(|j| match j {
            json::JsonValue::Object(j) => Some(j),
            _ => None,
        })
        .unwrap_or(None);

    // Register Benchmarks
    let benchmarks: Vec<String> = if let Some(data) = &json_data {
        let mut bb = vec![];
        match data.get("benchmarks") {
            Some(json::JsonValue::Array(vals)) => {
                for val in vals {
                    let name = match val {
                        json::JsonValue::Short(name) => name.as_str(),
                        json::JsonValue::String(name) => name,
                        _ => continue,
                    };
                    bb.push(name.into());
                }
            }
            _ => (),
        }
        bb
    } else {
        vec![]
    };
    rs.push(format!(
        "pub const BENCHMARKS: [&str; {}] = {:?};",
        benchmarks.len(),
        benchmarks
    ));

    // Register Simulators
    let simulators: Vec<String> = if let Some(data) = &json_data {
        let mut ss = vec![];
        match data.index("simulators") {
            json::JsonValue::Array(vals) => {
                for val in vals {
                    let name = match val {
                        json::JsonValue::Short(name) => name.as_str(),
                        json::JsonValue::String(name) => name,
                        _ => continue,
                    };
                    ss.push(name.into());
                }
            }
            _ => (),
        }
        ss
    } else {
        vec![]
    };
    rs.push(format!(
        "pub const SIMULATORS: [&str; {}] = {:?};",
        simulators.len(),
        simulators
    ));

    let mut qubits: Vec<usize> = vec![];
    let mut args: Vec<usize> = vec![];
    let mut benchmark_qubits: Vec<Vec<usize>> = vec![];
    let mut benchmark_args: Vec<Vec<usize>> = vec![];
    let mut simulator_qubits: Vec<Vec<Vec<usize>>> = vec![];
    let mut simulator_args: Vec<Vec<Vec<usize>>> = vec![];
    benchmark_qubits.resize(benchmarks.len(), vec![]);
    benchmark_args.resize(benchmarks.len(), vec![]);
    simulator_qubits.resize(benchmarks.len(), vec![]);
    simulator_args.resize(benchmarks.len(), vec![]);
    for (qubits, args) in simulator_qubits.iter_mut().zip(simulator_args.iter_mut()) {
        qubits.resize(simulators.len(), vec![]);
        args.resize(simulators.len(), vec![]);
    }

    if let Some(data) = json_data
        && let Some(JsonValue::Object(parameter_data)) = data.get("parameters")
    {
        maybe_push_parameters(
            &mut qubits,
            &mut args,
            &mut benchmark_qubits,
            &mut benchmark_args,
            &mut simulator_qubits,
            &mut simulator_args,
            parameter_data,
            &benchmarks,
            &simulators,
        );
    }

    if qubits.is_empty() {
        qubits = env::var("QUBITS").format(vec![2, 4, 6, 8]);
    }

    if args.is_empty() {
        args = env::var("ARGS").format(vec![1]);
    }

    rs.push(format!("pub const QUBITS: [usize; {}] = {:?};", qubits.len(), qubits));
    rs.push(format!("pub const ARGS: [usize; {}] = {:?};", args.len(), args));
    rs.push(format!("pub const BENCHMARK_QUBITS: [&[usize]; {}] = [", benchmarks.len()));
    for bench in benchmark_qubits.iter() {
        rs.push(format!("    &{:?},", bench));
    }
    rs.push("];".into());
    rs.push(format!("pub const BENCHMARK_ARGS: [&[usize]; {}] = [", benchmarks.len()));
    for bench in benchmark_args.iter() {
        rs.push(format!("    &{:?},", bench));
    }
    rs.push("];".into());
    rs.push(format!("pub const SIMULATOR_QUBITS: [[&[usize]; {}]; {}] = [", simulators.len(), benchmarks.len()));
    for bench in simulator_qubits.iter() {
        rs.push("    [".into());
        for sim in bench {
            rs.push(format!("        &{:?},", sim));
        }
        rs.push("    ],".into());
    }
    rs.push("];".into());
    rs.push(format!("pub const SIMULATOR_ARGS: [[&[usize]; {}]; {}] = [", simulators.len(), benchmarks.len()));
    for bench in simulator_args.iter() {
        rs.push("    [".into());
        for sim in bench {
            rs.push(format!("        &{:?},", sim));
        }
        rs.push("    ],".into());
    }
    rs.push("];".into());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_dir.join("bench_consts.rs"), rs.join("\n"))
        .unwrap_or_else(|e| eprintln!("error: {}", e));
}

fn filtered<T, U, F: FnOnce(T) -> Option<U>>(val: Option<T>, f: F) -> Option<U> {
    match val {
        Some(v) => f(v),
        None => None,
    }
}

fn string_filter(val: &JsonValue) -> Option<&str> {
    match val {
        JsonValue::Short(short) => Some(short.as_str()),
        JsonValue::String(s) => Some(s),
        _ => None,
    }
}

fn object_filter(val: &JsonValue) -> Option<&Object> {
    match val {
        JsonValue::Object(obj) => Some(obj),
        _ => None,
    }
}

fn maybe_push_parameters(
    out_qubits: &mut Vec<usize>,
    out_args: &mut Vec<usize>,
    bench_out_qubits: &mut Vec<Vec<usize>>,
    bench_out_args: &mut Vec<Vec<usize>>,
    sim_out_qubits: &mut Vec<Vec<Vec<usize>>>,
    sim_out_args: &mut Vec<Vec<Vec<usize>>>,
    obj: &Object,
    benchmarks: &[String],
    simulators: &[String],
) {
    maybe_push_item(out_qubits, obj, "qubits");
    maybe_push_item(out_args, obj, "args");

    let Some(JsonValue::Array(bench_array)) = obj.get("benchmarks") else {
        return;
    };

    for bench_data in bench_array.iter().filter_map(object_filter) {
        maybe_push_bench(
            bench_out_qubits,
            bench_out_args,
            sim_out_qubits,
            sim_out_args,
            bench_data,
            &benchmarks,
            &simulators,
        );
    }
}

fn maybe_push_bench(
    out_qubits: &mut Vec<Vec<usize>>,
    out_args: &mut Vec<Vec<usize>>,
    sim_out_qubits: &mut Vec<Vec<Vec<usize>>>,
    sim_out_args: &mut Vec<Vec<Vec<usize>>>,
    obj: &Object,
    benchmarks: &[String],
    simulators: &[String],
) {
    let Some(bench_name) = filtered(obj.get("name"), string_filter) else {
        return;
    };
    let Some(bench_i) = benchmarks.iter().position(|s| bench_name == s) else {
        return;
    };

    if out_qubits[bench_i].is_empty() {
        maybe_push_item(&mut out_qubits[bench_i], obj, "qubits");
    }
    if out_args[bench_i].is_empty() {
        maybe_push_item(&mut out_args[bench_i], obj, "args");
    }

    let Some(JsonValue::Array(sim_array)) = obj.get("simulators") else {
        return;
    };

    for sim_data in sim_array.iter().filter_map(object_filter) {
        maybe_push_sim(
            &mut sim_out_qubits[bench_i],
            &mut sim_out_args[bench_i],
            sim_data,
            &simulators,
        );
    }
}

fn maybe_push_sim(
    out_qubits: &mut Vec<Vec<usize>>,
    out_args: &mut Vec<Vec<usize>>,
    obj: &Object,
    simulators: &[String],
) {
    let Some(sim_name) = filtered(obj.get("name"), string_filter) else {
        return;
    };
    let Some(sim_i) = simulators.iter().position(|s| sim_name == s) else {
        return;
    };

    if out_qubits[sim_i].is_empty() {
        maybe_push_item(&mut out_qubits[sim_i], obj, "qubits");
    }
    if out_args[sim_i].is_empty() {
        maybe_push_item(&mut out_args[sim_i], obj, "args");
    }
}

fn maybe_push_item(out: &mut Vec<usize>, obj: &Object, item: &str) {
    let vals = match obj.get(item) {
        Some(JsonValue::Array(vals)) => vals,
        _ => &Vec::with_capacity(0),
    };
    maybe_push_array(out, vals);
}

fn maybe_push_array(out: &mut Vec<usize>, vals: &[JsonValue]) {
    for val in vals {
        maybe_push_value(out, val);
    }
}

fn maybe_push_value(out: &mut Vec<usize>, val: &JsonValue) {
    match val {
        JsonValue::Number(number) => maybe_push_number(out, number),
        _ => {}
    }
}

fn maybe_push_number(out: &mut Vec<usize>, val: &Number) {
    match val.as_fixed_point_u64(0) {
        Some(u) => out.push(u as usize),
        None => {}
    }
}

trait Format<Out: fmt::Debug> {
    fn format(self, fallback: Out) -> Out;
    fn use_fallback<E: Error>(error: E, fallback: Out) -> Out {
        eprintln!("warn: Fall back to {:?}, due to error: {}", fallback, error);
        fallback
    }
}

impl Format<Vec<usize>> for String {
    fn format(self, fallback: Vec<usize>) -> Vec<usize> {
        self.split(',')
            .map(|s| s.trim().parse())
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|e| Self::use_fallback(e, fallback))
    }
}
impl<T: Format<O>, O: fmt::Debug, E: Error> Format<O> for Result<T, E> {
    fn format(self, fallback: O) -> O {
        match self {
            Ok(v) => v.format(fallback),
            Err(e) => Self::use_fallback(e, fallback),
        }
    }
}
