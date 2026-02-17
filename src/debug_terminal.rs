mod arguments;
mod command;
mod parse;

pub use arguments::*;
pub use command::*;

use crate::{
    Circuit, DebugSimulator, SimpleSimulator,
    debug_terminal::parse::{ParseError, Token, into_tokens},
};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    execute, queue,
    style::{self, Attributes, Color, ContentStyle, StyledContent},
    terminal,
};
use std::{
    fmt::{Display, write},
    io::{self, Read, Write},
    ops::RangeInclusive,
    str::FromStr,
};

pub struct DebugTerminal {
    simulator: DebugSimulator,
}

impl DebugTerminal {
    pub fn new(circuit: Circuit) -> Result<Self, <DebugSimulator as SimpleSimulator>::E> {
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
                Command::Help(help_args) => Self::print(&mut stdout, &"Help")?,
                Command::Continue(continue_args) => Self::print(&mut stdout, &"Continue")?,
                Command::Next(next_args) => Self::print(&mut stdout, &"Next")?,
                Command::Break(break_args) => Self::print(&mut stdout, &"Break")?,
                Command::Delete(delete_args) => Self::print(&mut stdout, &"Delete")?,
                Command::Disable(disable_args) => Self::print(&mut stdout, &"Disable")?,
                Command::State(state_args) => Self::print(&mut stdout, &"State")?,
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
}
