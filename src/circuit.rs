use openqasm::{GateWriter, Linearize, Parser, Program, SourceCache, parser::ParseError};

use crate::Instruction;

const LINEARIZER_DEPTH: usize = 99999;

pub struct Circuit {
    pub instructions: Vec<Instruction>,
    pub n_qubits: usize,
}

enum QasmParseError {
    ParseError,
    TypeError
}

impl From<ParseError> for QasmParseError {
    fn from(e: ParseError) -> Self {
        QasmParseError::ParseError(e)
    }
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Circuit {
            instructions: Vec::<Instruction>::default(),
            n_qubits,
        }
    }

    pub fn from_qasm_string(qasm: String) -> Result<Self, Vec<QasmParseError>> {
        let mut cache: SourceCache = SourceCache::new();
        let mut parser: Parser = Parser::new(&mut cache);
        parser.parse_source(qasm, None);
        let program: Program = parser.done()?;
        //program.type_check()?;
        let mut l = Linearize::new(CircuitWriter(), LINEARIZER_DEPTH);
    }

    pub fn from_qasm_file(filename: &str) -> Self {
        unimplemented!()
    }

    // Functions for extending with a single gate below

    pub fn x(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::X(target));
        self
    }

    pub fn y(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Y(target));
        self
    }

    pub fn z(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::Z(target));
        self
    }

    pub fn hadamard(mut self, target: usize) -> Self {
        self.instructions.push(Instruction::H(target));
        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.instructions.push(Instruction::CNOT(control, target));
        self
    }
}

// Internal mechanisms for parsing QASM and converting to a circuit does not need to be pub
struct CircuitWriter();
impl GateWriter for CircuitWriter {
    // TODO: Investigate if a more specific error type is needed here, or if Infallible is sufficient
    type Error = std::convert::Infallible;

    fn initialize(
        &mut self,
        qubits: &[openqasm::Symbol],
        bits: &[openqasm::Symbol],
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn write_cx(&mut self, copy: usize, xor: usize) -> Result<(), Self::Error> {
        todo!()
    }

    fn write_u(
        &mut self,
        theta: openqasm::Value,
        phi: openqasm::Value,
        lambda: openqasm::Value,
        reg: usize,
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn write_opaque(
        &mut self,
        name: &openqasm::Symbol,
        params: &[openqasm::Value],
        args: &[usize],
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn write_barrier(&mut self, regs: &[usize]) -> Result<(), Self::Error> {
        todo!()
    }

    fn write_measure(&mut self, from: usize, to: usize) -> Result<(), Self::Error> {
        todo!()
    }

    fn write_reset(&mut self, reg: usize) -> Result<(), Self::Error> {
        todo!()
    }

    fn start_conditional(&mut self, reg: usize, count: usize, val: u64) -> Result<(), Self::Error> {
        todo!()
    }

    fn end_conditional(&mut self) -> Result<(), Self::Error> {
        todo!()
    }
}
