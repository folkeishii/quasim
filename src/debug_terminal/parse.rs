pub type Token<'a> = &'a str;
pub type TokenIterator<'a> = std::str::Split<'a, char>;

#[macro_export]
/// thiserror freaking out about implementing `From<<usize as FromStr>::Err>`
/// for `ParseError`.
macro_rules! parse_usize {
    ($string:expr) => {
        $string
            .parse()
            .map_err(|_| ParseError::ExpectedUnsigned($string.into()))
    };
}

pub fn into_tokens(input: &str, seperator: char) -> TokenIterator<'_> {
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
