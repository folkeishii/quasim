use crate::{DebugTerminal, PrevArgs};
use std::io;
use std::io::Write;

impl DebugTerminal {
    pub fn prev<W: Write>(&mut self, stdout: &mut W, prev_args: PrevArgs) -> io::Result<()> {
        match prev_args {
            PrevArgs::Back => {
                if self.simulator.step_backwards().is_none() {
                    Self::error(stdout, &"Already at the beginning")?
                }
            }
            PrevArgs::Count(n) => {
                for _ in 0..n {
                    if self.simulator.step_backwards().is_none() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}
