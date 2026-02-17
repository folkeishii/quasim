use crate::{DebugTerminal, StateArgs};
use std::io;
use std::io::Write;

impl DebugTerminal {
    pub fn print_state<W: Write>(
        &mut self,
        stdout: &mut W,
        state_args: StateArgs,
    ) -> io::Result<()> {
        let current_state = self.simulator.current_state();
        let state_size = current_state.len() - 1;

        match &state_args {
            StateArgs::All => {}
            StateArgs::Range(r) => {
                if *r.start() > state_size {
                    Self::error(
                        stdout,
                        &format!("{} is outside the state size of {}", r.start(), state_size),
                    )?;
                    return Ok(());
                } else if *r.end() > state_size {
                    Self::error(
                        stdout,
                        &format!("{} is outside the state size of {}", r.end(), state_size),
                    )?;
                    return Ok(());
                }
            }
            StateArgs::Multiple(ms) => {
                for m in ms {
                    if *m > state_size {
                        Self::error(
                            stdout,
                            &format!("{} is outside the state size of {}", m, state_size),
                        )?;
                        return Ok(());
                    }
                }
            }
            StateArgs::Single(s) => {
                if *s > state_size {
                    Self::error(
                        stdout,
                        &format!("{} is outside the state size of {}", s, state_size),
                    )?;
                    return Ok(());
                }
            }
        }

        let to_show: Vec<IndexedState> = match state_args {
            StateArgs::All => current_state
                .into_iter()
                .enumerate()
                .map(|(i, f)| IndexedState {
                    index: i,
                    state: f.to_string(),
                })
                .collect(),
            StateArgs::Range(r) => r
                .map(|r| IndexedState {
                    index: r,
                    state: current_state
                        .get(r)
                        .expect("How did we get here? Just to suffer?")
                        .to_string(),
                })
                .collect(),
            StateArgs::Multiple(ms) => ms
                .into_iter()
                .map(|m| IndexedState {
                    index: m,
                    state: current_state
                        .get(m)
                        .expect("How did we get here? Just to suffer?")
                        .to_string(),
                })
                .collect(),
            StateArgs::Single(s) => {
                vec![IndexedState {
                    index: s,
                    state: current_state
                        .get(s)
                        .expect("How did we get here? Just to suffer?")
                        .to_string(),
                }]
            }
        };

        if to_show.len() == 1 {
            let state = to_show.first().unwrap();
            Self::print(stdout, &format!("[ {} ] {}\n", state.state, state.index))?;
            return Ok(());
        }

        let state_width = to_show.iter().map(|i| i.state.len()).max().unwrap_or(1);

        Self::print(stdout, &format!("┌ {} ┐\n", " ".repeat(state_width)))?;
        for s in to_show {
            Self::print(
                stdout,
                &format!("│ {: >state_width$} │ {}\n", s.state, s.index),
            )?;
        }
        Self::print(stdout, &format!("└ {} ┘\n", " ".repeat(state_width)))?;

        Ok(())
    }
}

struct IndexedState {
    index: usize,
    state: String,
}
