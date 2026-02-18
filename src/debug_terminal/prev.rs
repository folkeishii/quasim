use crate::{DebugTerminal, PrevArgs};
use std::io;
use std::io::Write;

impl DebugTerminal {
    pub fn prev<W: Write>(&mut self, stdout: &mut W, prev_args: PrevArgs) -> io::Result<()> {
        let step_count = match prev_args {
            PrevArgs::Back => 1,
            PrevArgs::Count(n) => n,
        };

        for _ in 0..step_count {
            if self.simulator.step_backwards().is_none() {
                Self::error(stdout, &"Already at the beginning")?;
                break;
            }
        }

        Ok(())
    }
}
