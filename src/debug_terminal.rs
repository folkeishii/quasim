mod arguments;
mod command;
mod parse;
mod state;

pub use arguments::*;
pub use command::*;

use crate::ext::collapse;
use crate::{
    circuit::Circuit,
    debug_simulator::DebugSimulator,
    debug_terminal::parse::into_tokens,
    simulator::{BuildSimulator, DebuggableSimulator, DoubleEndedSimulator},
};
use crossterm::{
    execute,
    style::{self, Attributes, Color, ContentStyle, StyledContent},
};
use std::collections::HashMap;
use std::ops::Div;
use std::{
    fmt::Display,
    io::{self, Write},
};

pub struct DebugTerminal {
    simulator: DebugSimulator,
}

impl DebugTerminal {
    pub fn new(circuit: Circuit) -> Result<Self, <DebugSimulator as BuildSimulator>::E> {
        Ok(Self {
            simulator: DebugSimulator::build(circuit)?,
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
                Command::Next(_next_args) => Self::print(&mut stdout, &"Next")?,
                Command::Previous(prev_args) => self.prev(&mut stdout, prev_args)?,
                Command::Break(_break_args) => Self::print(&mut stdout, &"Break")?,
                Command::Delete(_delete_args) => Self::print(&mut stdout, &"Delete")?,
                Command::Disable(_disable_args) => Self::print(&mut stdout, &"Disable")?,
                Command::State(state_args) => self.print_state(&mut stdout, state_args)?,
                Command::Collapse(collapse_args) => {
                    self.collapse_state(&mut stdout, collapse_args)?
                }
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

    fn collapse_state<W: Write>(
        &mut self,
        stdout: &mut W,
        collapse_args: CollapseArgs,
    ) -> io::Result<()> {
        let state = self.simulator.current_state();
        let mut count_map: HashMap<usize, usize> = HashMap::new();
        let count = match collapse_args {
            CollapseArgs::Collapse => 1,
            CollapseArgs::Count(n) => n,
        };

        for _ in 0..count {
            let collapsed = collapse(state.as_ref());
            *count_map.entry(collapsed).or_insert(0) += 1;
        }

        let mut counts = count_map.iter().collect::<Vec<_>>();
        counts.sort_by(|a, b| b.1.cmp(&a.1));

        let max_index = counts.iter().map(|i| i.0).max().unwrap_or(&0);
        let decimal_width = (max_index.checked_ilog10().unwrap_or(0) + 1) as usize;
        let binary_width = (max_index.checked_ilog2().unwrap_or(0) + 1) as usize;

        let max_count = counts.iter().map(|i| i.1).max().unwrap_or(&1);
        let max_width = (max_count.checked_ilog10().unwrap_or(0) + 1) as usize;

        let mut count_peekable = counts.iter().peekable();
        while let Some(c) = count_peekable.next() {
            Self::print(
                stdout,
                &format!(
                    "{: >decimal_width$} ({:0binary_width$b}) - {: >max_width$} ({: >2}%)",
                    c.0,
                    c.0,
                    c.1,
                    ((*c.1 as f64).div(count as f64) * 100.0).round(),
                ),
            )?;

            if count_peekable.peek().is_some() {
                Self::print(stdout, &"\n")?;
            }
        }

        Ok(())
    }
}
