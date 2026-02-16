use crate::{Circuit, DebugSimulator, SimpleSimulator};
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

pub type Token<'a> = &'a str;
pub type TokenIterator<'a> = std::str::Split<'a, char>;

/// thiserror freaking out about implementing `From<<usize as FromStr>::Err>`
/// for `ParseError`.
macro_rules! parse_usize {
    ($string:expr) => {
        $string
            .parse()
            .map_err(|_| ParseError::ExpectedUnsigned($string.into()))
    };
}

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

enum TerminalEvent {
    Echo(char),
    Enter,
}

#[derive(Debug, Clone)]
/// Based on GDB commands
enum Command {
    /// TODO
    ///
    /// Quits the debugger
    ///
    /// Usage: (quit can be substituted with `q`)
    /// quit    # Quits the debugger
    Quit,

    /// TODO
    ///
    /// Outputs commands and their arguments
    ///
    /// Should be able to specify a specific argument
    ///
    /// Usage: (help can be substituted with `h`)
    /// help            # Print manual
    /// help [command]  # Print manual for command
    Help(HelpArgs),

    /// TODO
    ///
    /// Executes a set number of gates.
    ///
    /// Usage: (continue can be substituted with `c`)
    /// continue            # Continue until next breakpoint
    /// continue [count]    # Continue and skip `count` number of breakpoints
    /// continue ignore     # Continue and ignore all breakpoints
    /// run                 # Continue and ignore all breakpoints
    Continue(ContinueArgs),

    /// TODO
    ///
    /// Executes a set number of gates.
    ///
    /// Usage: (next can be substituted with `n`)
    /// next            # Execute the next gate
    /// next [count]    # Executes next `count` times
    Next(NextArgs),

    /// TODO
    ///
    /// Creates a breakpoint or enables a disabled breakpoint
    ///
    /// Usage:
    /// break [gate index...]     # Create or enable breakpoints at indices
    /// enable [gate index...]    # Enable a disabled breakpoints at indices
    Break(BreakArgs),

    /// TODO
    ///
    /// Deletes an existing breakpoint
    ///
    /// Error message if breakpoint does not exist
    ///
    /// Usage:
    /// delete [gate index...]    # Delete breakpoints at indices
    Delete(DeleteArgs),

    /// TODO
    ///
    /// Disables an existing breakpoint
    ///
    /// Error message if breakpoint does not exist
    ///
    /// Usage:
    /// disable [gate index...]    # Disable breakpoints at indices
    Disable(DisableArgs),

    /// TODO
    ///
    /// Gives information about specified state or
    /// breakpoint
    ///
    /// Usage:
    /// state                # Get all states (if possible)
    /// state [states...]    # Get specific states
    /// state state1..state2 # Get a range of states (if possible)
    State(StateArgs),
}
impl Command {
    pub fn parse_tokens(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let mut tokens = tokens;
        let command = CommandIdent::parse_command(&mut tokens)?;
        Ok(match command {
            CommandIdent::Help => Command::Help(HelpArgs::parse_arguments(tokens)?),
            CommandIdent::Continue => Command::Continue(ContinueArgs::parse_arguments(tokens)?),
            CommandIdent::Run => Command::Continue(ContinueArgs::parse_arguments(tokens)?),
            CommandIdent::Next => Command::Next(NextArgs::parse_arguments(tokens)?),
            CommandIdent::Break => Command::Break(BreakArgs::parse_arguments(tokens)?),
            CommandIdent::Enable => Command::Break(BreakArgs::parse_enable_arguments(tokens)?),
            CommandIdent::Delete => Command::Delete(DeleteArgs::parse_arguments(tokens)?),
            CommandIdent::Disable => Command::Disable(DisableArgs::parse_arguments(tokens)?),
            CommandIdent::State => Command::State(StateArgs::parse_arguments(tokens)?),
            CommandIdent::Quit => {
                if let Some(token) = tokens.next() {
                    return Err(ParseError::UnexpectedArgument(token.into()));
                } else {
                    Command::Quit
                }
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
/// Differs from Command since Identifier does
/// not carry any information about arguments.
/// Also some command identifiers simplify into
/// another command. For example `run` simplifies
/// to continuing and skipping all breakpoints
pub enum CommandIdent {
    /// quit or q
    Quit,
    /// help or h
    Help,
    /// continue or c
    Continue,
    /// run
    Run,
    /// next or n
    Next,
    /// break
    Break,
    /// enable
    Enable,
    /// delete
    Delete,
    /// disable
    Disable,
    /// state
    State,
}
impl CommandIdent {
    pub fn parse_command(tokens: &mut TokenIterator<'_>) -> ParseResult<Self> {
        let token = tokens
            .next()
            .ok_or(ParseError::ExpectedCommand("Nothing".into()))?;
        Self::try_from(token)
    }

    pub fn to_string(&self) -> String {
        match self {
            CommandIdent::Quit => "quit".into(),
            CommandIdent::Help => "help".into(),
            CommandIdent::Continue => "continue".into(),
            CommandIdent::Run => "run".into(),
            CommandIdent::Next => "next".into(),
            CommandIdent::Break => "break".into(),
            CommandIdent::Enable => "enable".into(),
            CommandIdent::Delete => "delete".into(),
            CommandIdent::Disable => "disable".into(),
            CommandIdent::State => "state".into(),
        }
    }
}
impl TryFrom<Token<'_>> for CommandIdent {
    type Error = ParseError;

    fn try_from(value: Token) -> ParseResult<Self> {
        match value {
            "quit" | "q" => Ok(Self::Quit),
            "help" | "h" => Ok(Self::Help),
            "continue" | "c" => Ok(Self::Continue),
            "run" => Ok(Self::Run),
            "next" | "n" => Ok(Self::Next),
            "break" => Ok(Self::Break),
            "enable" => Ok(Self::Enable),
            "delete" => Ok(Self::Delete),
            "disable" => Ok(Self::Disable),
            "state" => Ok(Self::State),
            _ => Err(ParseError::ExpectedCommand(value.into())),
        }
    }
}
impl Display for CommandIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string())
    }
}

#[derive(Debug, Clone, Copy)]
enum HelpArgs {
    All,
    Command(CommandIdent),
}
impl HelpArgs {
    pub fn parse_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let mut tokens = tokens;
        let arg = if let Some(token) = tokens.next() {
            token
        } else {
            return Ok(HelpArgs::All);
        };

        if let Some(token) = tokens.next() {
            return Err(ParseError::UnexpectedArgument(token.into()));
        }

        Ok(HelpArgs::Command(arg.try_into()?))
    }
}

#[derive(Debug, Clone, Copy)]
enum ContinueArgs {
    IgnoreBreak,
    UntilBreak,
    SkipBreaks(usize),
}
impl ContinueArgs {
    pub fn parse_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let mut tokens = tokens;
        let arg = if let Some(token) = tokens.next() {
            token
        } else {
            return Ok(ContinueArgs::UntilBreak);
        };

        if let Some(token) = tokens.next() {
            return Err(ParseError::UnexpectedArgument(token.into()));
        }

        if arg == "ignore" {
            return Ok(ContinueArgs::IgnoreBreak);
        }

        Ok(ContinueArgs::SkipBreaks(parse_usize!(arg)?))
    }

    pub fn parse_run_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let mut tokens = tokens;
        if let Some(token) = tokens.next() {
            Err(ParseError::UnexpectedArgument(token.into()))
        } else {
            Ok(ContinueArgs::IgnoreBreak)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NextArgs {
    Step,
    Count(usize),
}
impl NextArgs {
    pub fn parse_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let mut tokens = tokens;
        let arg = if let Some(token) = tokens.next() {
            token
        } else {
            return Ok(NextArgs::Step);
        };

        if let Some(token) = tokens.next() {
            return Err(ParseError::UnexpectedArgument(token.into()));
        }

        Ok(NextArgs::Count(parse_usize!(arg)?))
    }
}

#[derive(Debug, Clone)]
enum BreakArgs {
    GateIndices(Vec<usize>),
    GateIndicesEnable(Vec<usize>),
    // Leaving room in case we decide to add
    // more argument types
}
impl BreakArgs {
    pub fn parse_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        Ok(BreakArgs::GateIndices(Self::parse_inner(tokens)?))
    }
    pub fn parse_enable_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        Ok(BreakArgs::GateIndicesEnable(Self::parse_inner(tokens)?))
    }

    fn parse_inner(tokens: TokenIterator<'_>) -> ParseResult<Vec<usize>> {
        let indices = tokens
            .map(|token| parse_usize!(token))
            .collect::<ParseResult<Vec<_>>>()?;
        if indices.is_empty() {
            Err(ParseError::ExpectedArgument("Nothing".into()))
        } else {
            Ok(indices)
        }
    }
}

#[derive(Debug, Clone)]
enum DeleteArgs {
    GateIndices(Vec<usize>),
    // Leaving room in case we decide to add
    // more argument types
}
impl DeleteArgs {
    pub fn parse_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let indices = tokens
            .map(|token| parse_usize!(token))
            .collect::<ParseResult<Vec<_>>>()?;
        if indices.is_empty() {
            Err(ParseError::ExpectedArgument("Nothing".into()))
        } else {
            Ok(DeleteArgs::GateIndices(indices))
        }
    }
}

#[derive(Debug, Clone)]
enum DisableArgs {
    GateIndices(Vec<usize>),
    // Leaving room in case we decide to add
    // more argument types
}
impl DisableArgs {
    pub fn parse_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let indices = tokens
            .map(|token| parse_usize!(token))
            .collect::<ParseResult<Vec<_>>>()?;
        if indices.is_empty() {
            Err(ParseError::ExpectedArgument("Nothing".into()))
        } else {
            Ok(DisableArgs::GateIndices(indices))
        }
    }
}

#[derive(Debug, Clone)]
enum StateArgs {
    All,
    Range(RangeInclusive<usize>),
    Multiple(Vec<usize>),
    Single(usize),
}
impl StateArgs {
    pub fn parse_arguments(tokens: TokenIterator<'_>) -> ParseResult<Self> {
        let mut tokens = tokens;
        let arg1 = if let Some(token) = tokens.next() {
            token
        } else {
            return Ok(StateArgs::All);
        };

        let arg2 = if let Some(token) = tokens.next() {
            token
        } else {
            // Must be range or single
            return match arg1.split_once("..") {
                None => Ok(StateArgs::Single(parse_usize!(arg1)?)),
                Some((from, including)) => Ok(StateArgs::Range(
                    parse_usize!(from)?..=parse_usize!(including)?,
                )),
            };
        };

        // Must be multiple
        Ok(StateArgs::Multiple(
            [parse_usize!(arg1), parse_usize!(arg2)]
                .into_iter()
                .chain(tokens.map(|token| parse_usize!(token)))
                .collect::<ParseResult<Vec<_>>>()?,
        ))
    }
}

fn into_tokens(input: &str, seperator: char) -> TokenIterator<'_> {
    input.split(seperator)
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Expected command found \"{0}\"")]
    ExpectedCommand(String),
    #[error("Expected argument, found \"{0}\"")]
    ExpectedArgument(String),
    #[error("Unexpected argument \"{0}\"")]
    UnexpectedArgument(String),
    #[error("Expected unsigned integer found \"{0}\"")]
    ExpectedUnsigned(String),
}
pub type ParseResult<T> = Result<T, ParseError>;
