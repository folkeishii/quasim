mod arguments;
mod breakpoint;
mod command;
mod parse;
mod state;

pub use arguments::*;
pub use command::*;

use crate::{
    circuit::Circuit,
    debug_simulator::DebugSimulator,
    debug_terminal::{
        breakpoint::{BreakpointList, PEBreakpoint},
        parse::into_tokens,
    },
    simulator::{BuildSimulator, DebuggableSimulator, DoubleEndedSimulator},
};
use crossterm::{
    execute,
    style::{self, Attributes, Color, ContentStyle, StyledContent},
};
use std::{
    fmt::Display,
    io::{self, Write},
};

pub struct DebugTerminal {
    simulator: DebugSimulator,
    /// Sorted array of breakpoints
    /// i.e. Breakpoints are in order
    /// of gate index
    breakpoints: BreakpointList,
}

impl DebugTerminal {
    pub fn new(circuit: Circuit) -> Result<Self, <DebugSimulator as BuildSimulator>::E> {
        Ok(Self {
            simulator: DebugSimulator::build(circuit)?,
            breakpoints: Default::default(),
        })
    }

    pub fn run(&mut self) -> io::Result<()> {
        let mut stdout = io::stdout();
        let stdin = io::stdin();
        let mut input_buffer = String::default();

        loop {
            execute!(stdout, style::Print("\r\nqdb> "))?;
            input_buffer.clear();
            stdin.read_line(&mut input_buffer)?;
            if input_buffer.ends_with('\n') {
                input_buffer.pop();
                if input_buffer.ends_with('\r') {
                    input_buffer.pop();
                }
            }
            let command_tokens = into_tokens(&input_buffer, ' ');
            let command = match Command::parse_tokens(command_tokens) {
                Ok(c) => c,
                Err(e) => {
                    Self::error(&mut stdout, &e)?;
                    continue;
                }
            };

            match command {
                Command::Quit => break,
                Command::Help(_help_args) => Self::print(&mut stdout, &"Help")?,
                Command::Continue(_continue_args) => Self::print(&mut stdout, &"Continue")?,
                Command::Next(next_args) => self.next(&mut stdout, next_args)?,
                Command::Previous(prev_args) => self.prev(&mut stdout, prev_args)?,
                Command::Break(break_args) => self.handle_break(&mut stdout, &break_args)?,
                Command::Delete(_delete_args) => Self::print(&mut stdout, &"Delete")?,
                Command::Disable(_disable_args) => Self::print(&mut stdout, &"Disable")?,
                Command::State(state_args) => self.print_state(&mut stdout, state_args)?,
            }
        }
        Ok(())
    }

    fn print<W: Write, T: Display>(stdout: &mut W, output: &T) -> io::Result<()> {
        execute!(stdout, style::Print(output))
    }

    fn error<W: Write, T: Display>(stdout: &mut W, output: &T) -> io::Result<()> {
        execute!(
            stdout,
            style::PrintStyledContent(StyledContent::new(
                ContentStyle {
                    foreground_color: Some(Color::Red),
                    background_color: None,
                    underline_color: None,
                    attributes: Attributes::default()
                },
                "Error: "
            )),
            style::Print(output.to_string())
        )
    }

    fn next(&mut self, stdout: &mut io::Stdout, next_args: NextArgs) -> io::Result<()> {
        let step_checker = match next_args {
            NextArgs::Step => {
                let res = self.simulator.next().is_none();
                if !res {
                    Self::print(stdout, &"Stepped 1 time")?;
                }
                res
            }
            NextArgs::Count(n) => self.next_n_steps(stdout, n)?,
        };

        if !step_checker {
            Self::error(
                stdout,
                &format!("Cannot step further, end of circuit reached"),
            )?;
        }

        Ok(())
    }

    fn next_n_steps(&mut self, stdout: &mut io::Stdout, n: usize) -> io::Result<bool> {
        for i in 0..n {
            if self.simulator.next().is_none() {
                Self::error(stdout, &format!("Stepped {} time(s)", i))?;
                return Ok(false);
            }
        }
        Self::print(stdout, &format!("Stepped {} times", n))?;
        Ok(true)
    }

    fn prev<W: Write>(&mut self, stdout: &mut W, prev_args: PrevArgs) -> io::Result<()> {
        let step_count = match prev_args {
            PrevArgs::Back => 1,
            PrevArgs::Count(n) => n,
        };

        for _ in 0..step_count {
            if self.simulator.prev().is_none() {
                Self::error(stdout, &"Already at the beginning")?;
                break;
            }
        }

        Ok(())
    }

    fn handle_break<W: Write>(&mut self, stdout: &mut W, args: &BreakArgs) -> io::Result<()> {
        let (enable_only, gate_indices) = match args {
            BreakArgs::GateIndices(gate_indices) => (false, gate_indices),
            BreakArgs::GateIndicesEnable(gate_indices) => (true, gate_indices),
        };

        // Check that all indices are actual gates
        let gate_range = 0..self.simulator.instruction_count();
        for gate_index in gate_indices {
            if !gate_range.contains(gate_index) {
                Self::error(
                    stdout,
                    &format!(
                        "Unable to set or enable breakpoint at index {}, no gate at index",
                        gate_index
                    ),
                )?;
                return Ok(());
            }
        }

        // Insert or enable breakpoints
        for &gate_index in gate_indices {
            let status = if !enable_only {
                self.breakpoints.insert_or_enable(gate_index)
            } else if let Some(status) = self.breakpoints.enable(gate_index) {
                status
            } else {
                Self::error(
                    stdout,
                    &format!(
                        "Could not enable breakpoint at {}, breakpoint does not exist",
                        gate_index
                    ),
                )?;
                continue;
            };

            match status {
                PEBreakpoint::Enabled => {
                    Self::print(stdout, &format!("Enabled breakpoint at {}", gate_index))?
                }
                PEBreakpoint::Inserted => Self::print(
                    stdout,
                    &format!("Inserted new breakpoint at {}", gate_index),
                )?,
            }
        }

        Ok(())
    }

    fn print_state<W: Write>(&mut self, stdout: &mut W, state_args: StateArgs) -> io::Result<()> {
        let current_state = self.simulator.current_state();
        let to_show = match StateArgs::from_state(current_state, state_args) {
            Ok(v) => v,
            Err(e) => {
                Self::error(stdout, &e)?;
                return Ok(());
            }
        };

        if to_show.len() == 1 {
            let state = to_show.first().unwrap();
            Self::print(stdout, &format!("[ {} ] {}\n", state.state, state.index))?;
            return Ok(());
        }

        let state_width = to_show.iter().map(|i| i.state.len()).max().unwrap_or(1);
        let max_index = to_show.iter().map(|i| i.index).max().unwrap_or(1);
        let decimal_width = (max_index.checked_ilog10().unwrap_or(0) + 1) as usize;
        let binary_width = (max_index.checked_ilog2().unwrap_or(0) + 1) as usize;

        Self::print(stdout, &format!("┌ {} ┐\n", " ".repeat(state_width)))?;
        for s in to_show {
            Self::print(
                stdout,
                &format!(
                    "│ {: >state_width$} │ {: >decimal_width$} ({:0binary_width$b})\n",
                    s.state, s.index, s.index
                ),
            )?;
        }
        Self::print(stdout, &format!("└ {} ┘\n", " ".repeat(state_width)))?;

        Ok(())
    }
}
