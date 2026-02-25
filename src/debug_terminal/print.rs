use std::io::stdout;

use crossterm::{
    execute,
    style::{Color, ContentStyle, PrintStyledContent},
};
macro_rules! expand_print {
    ($cs:expr => $form:literal, $($disp:expr),+) => {
        expand_print!($cs => format!($form, $($disp),+))
    };
    ($cs:expr => $disp:expr) => {
        crossterm::style::PrintStyledContent(crossterm::style::StyledContent::new($cs, $disp))
    };
    ($form:literal, $($disp:expr),+) => {
        expand_print!(format!($form, $($disp),+))
    };
    ($disp:expr) => {
        crossterm::style::Print($disp)
    };
    // arg1 is displayed content
    // if arg2 is defined then arg2
    // is displayed content and arg1 is
    // contentstyle
    // ($arg1:expr $(, $arg2:expr)?) => {
    //     expand_print!($arg1 $(, $arg2)?)
    // };
}

// macro_rules! print {
//     ($w) => {
//         $crate::crossterm::execute!($w, "{} {}", $lol, $skib)
//     };
// }

#[test]
fn tprint() {
    let stdout = &mut stdout();
    execute!(
        stdout,
        expand_print!(
            ContentStyle {
                foreground_color: Some(Color::Blue),
                ..Default::default()
            } =>
            "lol\n"
        )
    )
    .unwrap();
    execute!(stdout, expand_print!("lol\n")).unwrap();
    execute!(stdout, expand_print!("lol\n")).unwrap();
    execute!(
        stdout,
        expand_print!(
            ContentStyle {
                foreground_color: Some(Color::Blue),
                ..Default::default()
            } =>
            "lol\n"
        )
    )
    .unwrap();
    execute!(stdout, expand_print!("{} {}\n", "lol", 3)).unwrap();
    execute!(
        stdout,
        expand_print!(
            ContentStyle {
                foreground_color: Some(Color::Red),
                ..Default::default()
            } =>
            "{} {}\n",
            "lol",
            3
        )
    )
    .unwrap();
}
