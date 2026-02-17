use std::ops::RangeInclusive;

use crate::{
    CommandIdent,
    debug_terminal::parse::{ParseError, ParseResult, TokenIterator},
    parse_usize,
};

#[derive(Debug, Clone, Copy)]
pub enum HelpArgs {
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
pub enum ContinueArgs {
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
pub enum NextArgs {
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
pub enum BreakArgs {
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
pub enum DeleteArgs {
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
pub enum DisableArgs {
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
pub enum StateArgs {
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
