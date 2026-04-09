use deutsch_jozsa::{FunctionType, find_function_type_quantum};
use quasim::sv_simulator::SVSimulatorDebugger;

fn main() {
    println!(
        "{}",
        find_function_type_quantum::<SVSimulatorDebugger>(8, FunctionType::Constant0)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f_constant(_: usize) -> bool {
        false
    }

    fn f_constant_2(_: usize) -> bool {
        true
    }

    fn f_balanced(x: usize, n: usize) -> bool {
        (x >> (n - 1)) == 1
    }

    fn f_balanced_2(x: usize) -> bool {
        x % 2 == 0
    }

    /// Check if a function is constant or balanced
    fn check_classic(f: impl Fn(usize) -> bool, n: usize) -> bool {
        let first = f(0);
        for i in 1..=(1 << (n - 1)) {
            if f(i) != first {
                return false;
            }
        }

        true
    }

    #[test]
    fn test_deutsch_jozsa() {
        for n in 2..6 {
            assert_eq!(
                check_classic(f_constant, n),
                find_function_type_quantum::<SVSimulatorDebugger>(n, FunctionType::Constant0)
            );
            assert_eq!(
                check_classic(f_constant_2, n),
                find_function_type_quantum::<SVSimulatorDebugger>(n, FunctionType::Constant1)
            );
            assert_eq!(
                check_classic(|c| f_balanced(c, n), n),
                find_function_type_quantum::<SVSimulatorDebugger>(n, FunctionType::Balanced)
            );
            assert_eq!(
                check_classic(f_balanced_2, n),
                find_function_type_quantum::<SVSimulatorDebugger>(n, FunctionType::Balanced)
            );
        }
    }
}
