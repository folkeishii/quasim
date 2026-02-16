use crate::{Circuit, DebugSimulator, SimpleSimulator};
use crossterm::{cursor, execute, queue, style, terminal};
use std::{
    fmt::Display,
    io::{self, Write},
    ops::RangeInclusive,
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
        // execute!(stdout, terminal::EnterAlternateScreen)?;

        // Raw mode allows most terminals to be used
        // in pretty much the same way.
        terminal::enable_raw_mode()?;

        for _ in 0..10 {
            // for line in MENU.split('\n') {
            //     queue!(w, style::Print(line), cursor::MoveToNextLine(1))?;
            // }

            // w.flush()?;

            // match read_char()? {
            //     '1' => test::cursor::run(w)?,
            //     '2' => test::color::run(w)?,
            //     '3' => test::attribute::run(w)?,
            //     '4' => test::event::run(w)?,
            //     '5' => test::synchronized_output::run(w)?,
            //     'q' => {
            //         execute!(w, cursor::SetCursorStyle::DefaultUserShape).unwrap();
            //         break;
            //     }
            //     _ => {}
            // };
            execute!(stdout, style::Print("test"), style::Print("\r\n"))?;
            // stdout.flush()?;
        }

        // execute!(
        //     w,
        //     style::ResetColor,
        //     cursor::Show,
        //     terminal::LeaveAlternateScreen
        // )?;

        terminal::disable_raw_mode()
    }

    fn queue_print<W: Write, T: Display>(stdout: &mut W, output: &T) -> io::Result<()> {
        terminal::disable_raw_mode()?;
        queue!(stdout, style::Print(output))?;
        terminal::enable_raw_mode()
    }

    fn queue_println<W: Write, T: Display>(stdout: &mut W, output: &T) -> io::Result<()> {
        Self::queue_print(stdout, output)?;
        queue!(stdout, style::Print("\r\n"))
    }
}

#[derive(Debug, Clone)]
/// Based on GDB commands
enum Command {
    /// TODO
    ///
    /// Quits the debugger
    Quit,

    /// TODO
    ///
    /// Outputs commands and their arguments
    ///
    /// Should be able to specify a specific argument
    Help(HelpArgs),

    /// TODO
    ///
    /// Executes a set number of gates.
    Continue(ContinueArgs),

    /// TODO
    ///
    /// Creates a breakpoint or enables a disabled breakpoint
    Break(BreakArgs),

    /// TODO
    ///
    /// Deletes an existing breakpoint
    ///
    /// Error message if breakpoint does not exist
    DeleteBreak(DeleteBreakArgs),

    /// TODO
    ///
    /// Disables an existing breakpoint
    ///
    /// Error message if breakpoint does not exist
    DisableBreak(DisableBreakArgs),

    /// TODO
    ///
    /// Gives information about specified state or
    /// breakpoint
    Info(InfoArgs),
}

#[derive(Debug, Clone, Copy)]
/// Differs from Command since Identifier does
/// not carry any information about arguments.
enum CommandIdent {
    Quit,
    Help,
    Continue,
    Break,
    DeleteBreak,
    DisableBreak,
    Info,
}
// impl CommandIdent {
//     pub fn parse_token()
// }

#[derive(Debug, Clone, Copy)]
enum HelpArgs {
    All,
    Command(CommandIdent),
}

#[derive(Debug, Clone, Copy)]
enum ContinueArgs {
    UntilBreak,
    Count(usize),
}

#[derive(Debug, Clone)]
enum BreakArgs {
    GateIndex(usize),
    // Leaving room in case we decide to add
    // more argument types
}

#[derive(Debug, Clone)]
enum DeleteBreakArgs {
    GateIndex(usize),
    // Leaving room in case we decide to add
    // more argument types
}

#[derive(Debug, Clone)]
enum DisableBreakArgs {
    GateIndex(usize),
    // Leaving room in case we decide to add
    // more argument types
}

#[derive(Debug, Clone)]
enum InfoArgs {
    /// The gate index of breakpoint
    Break(usize),
    State(StateInfo),
}

#[derive(Debug, Clone, Copy)]
/// The current value for specified state(s)
enum StateInfo {
    All,
    InclusiveRange(usize, usize),
    Single(usize),
}
