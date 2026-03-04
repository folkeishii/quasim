mod arguments;
mod command;
mod parse;
#[macro_use]
mod print;
mod show_circuit;
mod state;

pub use arguments::*;
pub use command::*;

use crate::{
    circuit::{
        Circuit,
        breakpoint::{BreakpointList, IEBreakpoint},
    },
    debug_simulator::DebugSimulator,
    debug_terminal::{parse::into_tokens, show_circuit::show_circuit},
    ext::collapse,
    simulator::{BuildSimulator, DebuggableSimulator, StoredCircuitSimulator},
};
use std::{
    io::{self, Write},
    ops::Div,
};

pub struct DebugTerminal<S = DebugSimulator> {
    simulator: S,
    /// Sorted array of breakpoints
    /// i.e. Breakpoints are in order
    /// of gate index
    breakpoints: BreakpointList,
}

impl<S> DebugTerminal<S>
where
    S: BuildSimulator + DebuggableSimulator + StoredCircuitSimulator,
{
    pub fn new(circuit: Circuit) -> Result<Self, <S as BuildSimulator>::E> {
        Ok(Self {
            simulator: S::build(circuit)?,
            breakpoints: Default::default(),
        })
    }

    pub fn run(&mut self) -> io::Result<()> {
        let mut stdout = io::stdout();
        let stdin = io::stdin();
        let mut input_buffer = String::default();

        loop {
            let current_step = match self.simulator.current_instruction() {
                None => "end".to_string(),
                Some((step, _)) => step.to_string(),
            };
            print!(stdout; "[{}] qdb> ", current_step)?;
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
                    errorln!(&mut stdout; &e)?;
                    continue;
                }
            };

            match command {
                Command::Quit => break,
                Command::Help(_help_args) => println!(&mut stdout; &"Help")?,
                Command::Continue(continue_args) => {
                    self.handle_continue(&mut stdout, &continue_args)?
                }
                Command::Next(next_args) => self.handle_next(&mut stdout, &next_args)?,
                Command::Previous(prev_args) => self.handle_prev(&mut stdout, &prev_args)?,
                Command::Break(break_args) => self.handle_break(&mut stdout, &break_args)?,
                Command::Delete(delete_args) => self.handle_delete(&mut stdout, &delete_args)?,
                Command::Disable(disable_args) => {
                    self.handle_disable(&mut stdout, &disable_args)?
                }
                Command::State(state_args) => self.handle_state(&mut stdout, &state_args)?,
                Command::Collapse(collapse_args) => {
                    self.handle_collapse(&mut stdout, &collapse_args)?
                }
                Command::Show(show_args) => self.handle_show(&mut stdout, &show_args)?,
            }
        }
        Ok(())
    }

    fn handle_continue(
        &mut self,
        stdout: &mut io::Stdout,
        continue_args: &ContinueArgs,
    ) -> io::Result<()> {
        match continue_args {
            ContinueArgs::UntilBreak => {
                let pc = self.simulator.current_instruction().as_ref().map(|(pc, _)| pc.pc()).unwrap_or(usize::MAX);
                let mut next_break = self.breakpoints.next_break(pc);
                while let Some(br) = next_break && !br.enabled() {
                    next_break = self.breakpoints.next_break(br.pc());
                }

                loop {
                    if self.simulator.next().is_none() {
                        println!(stdout; &"End of Circuit reached, continued until end")?;
                        return Ok(());
                    }

                    //Check if a breakpoint exists otherwise continue until end
                    let Some(next_break) = next_break else {
                        self.simulator.next();
                        continue;
                    };

                    //Check if the current index is the next break otherwise rerun the loop
                    let Some((instruction_index, _instruction)) =
                        self.simulator.current_instruction()
                    else {
                        continue;
                    };

                    if instruction_index.pc() == next_break.pc() {
                        println!(
                            stdout;
                            "Continued until breakpoint at index {}", next_break.pc()
                        )?;
                        return Ok(());
                    }
                }
            }
            ContinueArgs::SkipBreaks(n) => {
                let mut pc = self.simulator.current_instruction().as_ref().map(|(pc, _)| pc.pc()).unwrap_or(usize::MAX);
                let mut breakpoints_skipped = 0;
                loop {
                    let next_break = self.breakpoints.next_break(pc);

                    loop {
                        if self.simulator.next().is_none() {
                            println!(stdout; &"End of Circuit reached, continued until end")?;
                            return Ok(());
                        }

                        // If there are no more breaks to skip, continue until end
                        if next_break.is_none() {
                            println!(
                                stdout;
                                "Skipped {} breakpoints, continuing until end",
                                breakpoints_skipped
                            )?;
                        }

                        //Check if a breakpoint exists otherwise continue until end
                        let Some(next_break) = next_break else {
                            self.simulator.next();
                            continue;
                        };

                        //Check if the current index is the next break otherwise rerun the loop
                        let Some((instruction_index, _instruction)) =
                            self.simulator.current_instruction()
                        else {
                            continue;
                        };
                        pc = instruction_index.pc();

                        // If we have skipped the desired amount of breakpoints
                        if breakpoints_skipped == *n {
                            println!(
                                stdout;
                                "Skipped {} breakpoints, continued to index {}",
                                breakpoints_skipped, instruction_index
                            )?;
                            return Ok(());
                        }

                        if next_break.enabled() && instruction_index.pc() == next_break.pc() {
                            breakpoints_skipped += 1;
                            break;
                        }
                    }
                    self.simulator.next();
                }
            }
            ContinueArgs::IgnoreBreak => loop {
                if self.simulator.next().is_none() {
                    println!(stdout; &"End of Circuit reached, continued until end")?;
                    return Ok(());
                }
                self.simulator.next();
            },
        }
    }

    fn handle_next<W: Write>(&mut self, stdout: &mut W, next_args: &NextArgs) -> io::Result<()> {
        let step_count = match next_args {
            NextArgs::Step => 1,
            NextArgs::Count(n) => *n,
        };

        for i in 0..step_count {
            if self.simulator.next().is_none() {
                errorln!(
                    stdout;
                    "End of Circuit reached, stepped forward {} time(s)", i
                )?;
                return Ok(());
            }
        }
        println!(stdout; "Stepped forward {} time(s)", step_count)?;

        Ok(())
    }

    fn handle_prev<W: Write>(&mut self, stdout: &mut W, prev_args: &PrevArgs) -> io::Result<()> {
        let step_count = match prev_args {
            PrevArgs::Back => 1,
            PrevArgs::Count(n) => *n,
        };

        for _ in 0..step_count {
            if self.simulator.prev().is_none() {
                errorln!(stdout; "Already at the beginning")?;
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
        let gate_range = 0..self.simulator.circuit().instructions().len();
        for gate_index in gate_indices {
            if !gate_range.contains(gate_index) {
                errorln!(
                    stdout;
                    "Unable to set or enable breakpoint at index {}, no gate at index",
                    gate_index
                )?;
                return Ok(());
            }
        }

        // Insert or enable breakpoints
        for &gate_index in gate_indices {
            let status = if !enable_only {
                self.breakpoints.insert_or_enable(gate_index)
            } else if self.breakpoints.enable(gate_index) {
                IEBreakpoint::Enabled
            } else {
                errorln!(
                    stdout;
                    "Could not enable breakpoint at {}, breakpoint does not exist",
                    gate_index
                )?;
                continue;
            };

            match status {
                IEBreakpoint::Enabled => println!(stdout; "Enabled breakpoint at {}", gate_index)?,
                IEBreakpoint::Inserted => println!(
                    stdout;
                    "Inserted new breakpoint at {}", gate_index
                )?,
            }
        }

        Ok(())
    }

    fn handle_disable<W: Write>(&mut self, stdout: &mut W, args: &DisableArgs) -> io::Result<()> {
        let disable_indexes = match args {
            DisableArgs::GateIndices(indices) => indices,
        };

        let gate_range = 0..self.simulator.instruction_count();
        for gate_index in disable_indexes {
            if !gate_range.contains(gate_index) {
                errorln!(
                    stdout;
                    "Unable to disable breakpoint at index {}, no gate at index", gate_index
                )?;
                return Ok(());
            }
        }

        let mut disabled: Vec<String> = Vec::new();
        for &gate_index in disable_indexes {
            if !self.breakpoints.disable(gate_index) {
                errorln!(
                    stdout;
                    "Could not disable breakpoint at {}, breakpoint does not exist", gate_index
                )?;
                continue;
            }

            disabled.push(gate_index.to_string());
        }
        if !disabled.is_empty() {
            println!(
                stdout;
                "Disabled breakpoints {}", disabled.join(", ")
            )?;
        }

        Ok(())
    }

    fn handle_delete<W: Write>(&mut self, stdout: &mut W, args: &DeleteArgs) -> io::Result<()> {
        let delete_indexes = match args {
            DeleteArgs::GateIndices(i) => i,
        };

        let gate_range = 0..self.simulator.instruction_count();
        for gate_index in delete_indexes {
            if !gate_range.contains(gate_index) {
                errorln!(
                    stdout;
                    "Unable to delete breakpoint at index {}, no gate at index", gate_index
                )?;
                return Ok(());
            }
        }

        let mut deleted: Vec<String> = Vec::new();
        for &gate_index in delete_indexes {
            if !self.breakpoints.delete(gate_index) {
                errorln!(
                    stdout;
                    "Could not delete breakpoint at {}, breakpoint does not exist", gate_index
                )?;
                continue;
            }

            deleted.push(gate_index.to_string());
        }
        if !deleted.is_empty() {
            println!(
                stdout;
                "Deleted breakpoints {}", deleted.join(", ")
            )?;
        }

        Ok(())
    }

    fn handle_state<W: Write>(&mut self, stdout: &mut W, state_args: &StateArgs) -> io::Result<()> {
        let current_state = self.simulator.current_state();
        let to_show = match StateArgs::from_state(current_state, state_args) {
            Ok(v) => v,
            Err(e) => {
                errorln!(stdout; e)?;
                return Ok(());
            }
        };

        if to_show.len() == 1 {
            let state = to_show.first().unwrap();
            println!(stdout; "[ {} ] {}", state.state, state.index)?;
            return Ok(());
        }

        let state_width = to_show.iter().map(|i| i.state.len()).max().unwrap_or(1);
        let max_index = to_show.iter().map(|i| i.index).max().unwrap_or(1);
        let decimal_width = (max_index.checked_ilog10().unwrap_or(0) + 1) as usize;
        let binary_width = (max_index.checked_ilog2().unwrap_or(0) + 1) as usize;

        println!(stdout; "┌ {} ┐", " ".repeat(state_width))?;
        for s in to_show {
            println!(
                stdout;
                "│ {: >state_width$} │ {: >decimal_width$} ({:0binary_width$b})",
                s.state, s.index, s.index
            )?;
        }
        println!(stdout; "└ {} ┘", " ".repeat(state_width))?;

        Ok(())
    }

    fn handle_collapse<W: Write>(
        &mut self,
        stdout: &mut W,
        collapse_args: &CollapseArgs,
    ) -> io::Result<()> {
        let state = self.simulator.current_state();
        let mut count_map: Vec<(usize, usize)> = Vec::new();
        let count = match collapse_args {
            CollapseArgs::Collapse => 1,
            CollapseArgs::Count(n) => *n,
        };

        let mut max_state = 0; // Only used for formatting
        let mut max_count = 0; // Only used for formatting
        for _ in 0..count {
            let collapsed = collapse(state.as_ref());
            max_state = max_state.max(collapsed);
            let new_count = match count_map.binary_search_by_key(&collapsed, |(state, _)| *state) {
                // state already recorded
                Ok(index) => {
                    count_map[index].1 += 1;
                    count_map[index].1
                }
                // New state
                Err(index) => {
                    count_map.insert(index, (collapsed, 1));
                    count_map[index].1
                }
            };
            max_count = max_count.max(new_count)
        }

        // Resort by count
        count_map.sort_by(|a, b| b.1.cmp(&a.1));

        let decimal_width = (max_state.checked_ilog10().unwrap_or(0) + 1) as usize;
        let binary_width = (max_state.checked_ilog2().unwrap_or(0) + 1) as usize;

        let max_width = (max_count.checked_ilog10().unwrap_or(0) + 1) as usize;

        for c in count_map {
            println!(
                stdout;
                "{: >decimal_width$} ({:0binary_width$b}) - {: >max_width$} ({: >2}%)",
                c.0,
                c.0,
                c.1,
                ((c.1 as f64).div(count as f64) * 100.0).round()
            )?;
        }

        Ok(())
    }

    fn handle_show<W: Write>(&mut self, stdout: &mut W, show_args: &ShowArgs) -> io::Result<()> {
        match show_args {
            ShowArgs::Circuit => show_circuit(stdout, &self.simulator),
        }
    }
}
