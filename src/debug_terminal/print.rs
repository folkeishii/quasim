
use crossterm::{
    style::{Attributes, Color, ContentStyle},
};

pub const ERROR_STYLE: ContentStyle = ContentStyle {
    foreground_color: Some(Color::Red),
    background_color: None,
    underline_color: None,
    attributes: Attributes::none(),
};

macro_rules! print_args {
    ($cs:expr => $form:literal, $($disp:expr),+) => {
        print_args!($cs => format!($form, $($disp),+))
    };
    ($cs:expr => $disp:expr) => {
        crossterm::style::PrintStyledContent(crossterm::style::StyledContent::new($cs, $disp))
    };
    ($form:literal, $($disp:expr),+) => {
        print_args!(format!($form, $($disp),+))
    };
    ($disp:expr) => {
        crossterm::style::Print($disp)
    };
}

/// # Print
/// Implements print using crossterm, and with styling.
///
/// Styling is done via `crossterm::style::ContentStyle`
/// ```Rust
/// let red = ContentStyle {
///     foreground_color: Some(Color::Red),
///     background_color: None,
///     underline_color: None,
///     attributes: Attributes::none(),
/// };
/// let green = ContentStyle {
///     foreground_color: Some(Color::Green),
///     background_color: None,
///     underline_color: None,
///     attributes: Attributes::none(),
/// };
///
/// let stdout = &mut stdout();
/// // Print simple message
/// print!(stdout; "Hello World!\n")?;
///
/// // Print styled message
/// print!(stdout; green => "Green text!\n")?;
///
/// // Format message
/// print!(stdout; green => "{} {} {}!\n", 1, 2, 3)?;
///
/// // Style different parts of a message
/// print!(stdout; red => "{}!\n", 1; blue => "2!"; "3...")?;
/// ```
///
/// ## Syntax
/// The syntax is: `print!(stdout; Statement 1; Statement2; ...)`
///
/// And a statement consists of styling and the content.
///
/// If there is no styling then the statement is formatted like
macro_rules! print {
    ($w:expr; $($arg1:expr $(=> $arg2:expr)? $(, $disp:expr)*);+) => {
        crossterm::execute!($w, $(print_args!($arg1 $(=> $arg2)? $(, $disp)*)),+)
    };
}
macro_rules! println {
    ($w:expr; $($arg1:expr $(=> $arg2:expr)? $(, $disp:expr)*);+) => {
        print!($w; $($arg1 $(=> $arg2)? $(, $disp)*);+; "\r\n")
    };
}
macro_rules! error {
    ($w:expr; $($arg1:expr $(=> $arg2:expr)? $(, $disp:expr)*);+) => {
        print!($w;
            $crate::debug_terminal::print::ERROR_STYLE => "Error: ";
            $($arg1 $(=> $arg2)? $(, $disp)*);+
        )
    };
}
macro_rules! errorln {
    ($w:expr; $($arg1:expr $(=> $arg2:expr)? $(, $disp:expr)*);+) => {
        error!($w; $($arg1 $(=> $arg2)? $(, $disp)*);+; "\r\n")
    };
}
