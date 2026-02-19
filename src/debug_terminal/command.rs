use std::fmt::Display;

use crate::{
    debug_terminal::{parse::{ParseError, ParseResult, Token, TokenIterator},
    BreakArgs, ContinueArgs, DeleteArgs, DisableArgs, HelpArgs, NextArgs, StateArgs,
    },
};

#[derive(Debug, Clone)]
/// Based on GDB commands
pub enum Command {
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

    /// Print out the current state vector
    ///
    /// Usage:
    /// state                # Get all states (if possible)
    /// state state1         # Get a specific state (if possible)
    /// state [state1, ...]  # Get specific states (if possible)
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
