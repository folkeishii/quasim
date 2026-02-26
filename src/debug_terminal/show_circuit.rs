use std::{
    fmt::Display,
    io::{self, Write},
};

use crate::simulator::{DebuggableSimulator, StoredCircuitSimulator};

type Block = [[char; BLOCK_SIZE.width]; BLOCK_SIZE.height];
const BLOCK_SIZE: Size = Size {
    width: 3,
    height: 3,
};

struct Pos {
    lane: usize,
    column: usize,
}
struct Size {
    width: usize,
    height: usize,
}

enum Marker {
    Ket(usize),
    Gate { name: String, size: Size },
    SkipGate { width: usize },
    Measure,
}

fn ket<W: Write, T: Display>(write: &mut W, value: &T) -> io::Result<()> {
    print!(write; "|{}⟩", value)
}

fn print_circuit<W, S>(write: &mut W, simulator: &S, size: &Size) -> io::Result<()>
where
    W: Write,
    S: DebuggableSimulator + StoredCircuitSimulator,
{
    Ok(())
}

mod block {
    use std::{fmt::Display, marker::PhantomData};

    use crate::gate::{Gate, QBits};

    // https://en.wikipedia.org/wiki/Box-drawing_characters
    // https://www.alt-codes.net/bullet_alt_codes.php
    // https://www.compart.com/en/unicode/block/U+2B00
    use super::Block;

    #[derive(Debug, Clone, Copy)]
    pub struct MeasureGate(QBits);
    #[derive(Debug, Clone, Copy)]
    pub struct GateBlock<'a, G = (char, &'a Gate)>(G, PhantomData<&'a ()>);

    impl<'g> GateBlock<'g> {
        pub fn new<'a, T: Display>(name: T, gate: &'a Gate) -> GateBlock<'a, (T, &'a Gate)> {
            GateBlock::<'a, (T, &'a Gate)>((name, gate), Default::default())
        }

        pub fn measurement(targets: QBits) -> GateBlock<'g, MeasureGate> {
            GateBlock::<'g, MeasureGate>(MeasureGate(targets), Default::default())
        }

        /// Will panic if X, Y > 2
        pub fn get_char<const X: usize, const Y: usize>(&self, lane: usize) -> char {
            // if
            match (X, Y) {
                (1, 1) => *self.name(),
                (x, y) => gate::FL[y][x],
            }
        }
    }
    impl<'g, T: Display> GateBlock<'g, (T, &'g Gate)> {
        fn name(&self) -> &T {
            &self.0.0
        }
        fn gate(&self) -> &'g Gate {
            &self.0.1
        }

        fn lane_block(&self, lane: usize) -> Block {
            let targets = self.gate().get_target_bits().get_bitstring();
            let controls = self.gate().get_control_bits().get_bitstring();
            let t_and_c = targets | controls;
            let below_mask = usize::MAX << (lane + 1);
            let below_eq_mask = usize::MAX << lane;
            let above_mask = below_eq_mask ^ usize::MAX;
            let above_eq_mask = below_mask ^ usize::MAX;

            // Lane is outside targets and controls
            if t_and_c & below_mask == t_and_c || t_and_c & above_mask == t_and_c {
                track::OEOW
            }
            // Lane is below every target
            else if targets & above_mask == targets {
                // Lane has no controls below it
                if controls & below_mask == 0 {
                    // Lane must be on control
                    ctrl(track::NEOW)
                }
                // Lane has controls below it
                else {
                    // Lane is on control
                    if controls & (1 << lane) != 0 {
                        ctrl(track::NESW)
                    } else {
                        track::NESW_PASS
                    }
                }
            }
            // Lane is above every target
            else if targets & below_mask == targets {
                // Lane has no controls above it
                if controls & above_mask == 0 {
                    // Lane must be on control
                    ctrl(track::OESW)
                }
                // Lane has controls above it
                else {
                    // Lane is on control
                    if controls & (1 << lane) != 0 {
                        ctrl(track::NESW)
                    } else {
                        track::NESW_PASS
                    }
                }
            }
            // Lane is inside block
            else {
                // Lane is top and bot
                if targets & below_eq_mask & above_eq_mask == targets {
                    // Controls exists above and below
                    if controls & above_mask != 0 && controls & below_mask != 0 {
                        connect_south(connect_north(gate::FL))
                    }
                    // Controls exists above
                    else if controls & above_mask != 0 {
                        connect_north(gate::FL)
                    }
                    // Controls exists below
                    else if controls & below_mask != 0 {
                        connect_south(gate::FL)
                    } else {
                        gate::FL
                    }
                }
                // Lane is top
                else if targets & below_eq_mask == targets {
                    // Controls exist above
                    if controls & above_mask != 0 {
                        connect_north(gate::NN)
                    } else {
                        gate::NN
                    }
                }
                // Lane is bot
                else if targets & above_eq_mask == targets {
                    // Controls exist below
                    if controls & below_mask != 0 {
                        connect_south(gate::SS)
                    } else {
                        gate::SS
                    }
                }
                // Lane is on target
                else if targets & (1 << lane) != 0 {
                    connect_east(connect_west(gate::WE))
                }
                // Lane is control
                else if controls & (1 << lane) != 0 {
                    ctrl(gate::SK)
                } else {
                    gate::SK
                }
            }
        }
    }
    impl<'g> GateBlock<'g, MeasureGate> {
        pub fn get_char<const X: usize, const Y: usize>(&self, lane: usize) -> char {
            let targets = self.0.0.get_bitstring();
            if targets & (1 << lane) == 0 {
                return track::OEOW[Y][X];
            }

            match (X, Y) {
                (1, 1) => '∅',
                (x, y) => connect_west(connect_east(gate::FL))[y][x],
            }
        }
    }

    #[inline(always)]
    const fn ctrl(block: Block) -> Block {
        let mut block = block;
        block[1][1] = '⬤';
        block
    }

    #[inline(always)]
    const fn cinv(block: Block) -> Block {
        let mut block = block;
        block[1][1] = '⭘';
        block
    }

    #[inline(always)]
    const fn cnot(block: Block) -> Block {
        let mut block = block;
        block[1][1] = '⨁';
        block
    }

    #[inline(always)]
    const fn connect_north(block: Block) -> Block {
        let mut block = block;
        block[0][1] = '┴';
        block
    }
    #[inline(always)]
    const fn connect_east(block: Block) -> Block {
        let mut block = block;
        block[1][2] = '├';
        block
    }
    #[inline(always)]
    const fn connect_south(block: Block) -> Block {
        let mut block = block;
        block[2][1] = '┬';
        block
    }
    #[inline(always)]
    const fn connect_west(block: Block) -> Block {
        let mut block = block;
        block[1][0] = '┤';
        block
    }

    macro_rules! const_block {
        ($name:ident, $($($c:literal)*);*) => {
            pub const $name: Block = [
                $([$($c),*]),*
            ];
        };
    }

    pub mod track {
        use super::Block;

        const_block!(OOOO,
            ' ' ' ' ' ';
            ' ' ' ' ' ';
            ' ' ' ' ' '
        );
        const_block!(OOOW,
            ' ' ' ' ' ';
            '─' '─' ' ';
            ' ' ' ' ' '
        );
        const_block!(OOSO,
            ' ' ' ' ' ';
            ' ' '│' ' ';
            ' ' '│' ' '
        );
        const_block!(OOSW,
            ' ' ' ' ' ';
            '─' '┐' ' ';
            ' ' '│' ' '
        );
        const_block!(OEOO,
            ' ' ' ' ' ';
            ' ' '─' '─';
            ' ' ' ' ' '
        );
        const_block!(OEOW,
            ' ' ' ' ' ';
            '─' '─' '─';
            ' ' ' ' ' '
        );
        const_block!(OESO,
            ' ' ' ' ' ';
            ' ' '┌' '─';
            ' ' '│' ' '
        );
        const_block!(OESW,
            ' ' ' ' ' ';
            '─' '┬' '─';
            ' ' '│' ' '
        );
        const_block!(XOOO,
            ' ' '│' ' ';
            ' ' '│' ' ';
            ' ' ' ' ' '
        );
        const_block!(NOOW,
            ' ' '│' ' ';
            '─' '┘' ' ';
            ' ' ' ' ' '
        );
        const_block!(NOSO,
            ' ' '│' ' ';
            ' ' '│' ' ';
            ' ' '│' ' '
        );
        const_block!(NOSW,
            ' ' '│' ' ';
            '─' '┤' ' ';
            ' ' '│' ' '
        );
        const_block!(NEOO,
            ' ' '│' ' ';
            ' ' '└' '─';
            ' ' ' ' ' '
        );
        const_block!(NEOW,
            ' ' '│' ' ';
            '─' '┴' '─';
            ' ' ' ' ' '
        );
        const_block!(NESO,
            ' ' '│' ' ';
            ' ' '├' '─';
            ' ' '│' ' '
        );
        const_block!(NESW,
            ' ' '│' ' ';
            '─' '┼' '─';
            ' ' '│' ' '
        );

        const_block!(NESW_PASS,
            ' ' '│' ' ';
            '⬤' '│' '⬤';
            ' ' '│' ' '
        );
    }

    pub mod gate {
        use super::Block;

        const_block!(NL,
            ' ' ' ' ' ';
            ' ' ' ' ' ';
            ' ' ' ' ' '
        );
        const_block!(FL,
            '┌' '─' '┐';
            '│' ' ' '│';
            '└' '─' '┘'
        );
        const_block!(NW,
            '┌' '─' '─';
            '│' ' ' ' ';
            '│' ' ' ' '
        );
        const_block!(N,
            '─' '─' '─';
            ' ' ' ' ' ';
            ' ' ' ' ' '
        );
        const_block!(NN,
            '┌' '─' '┐';
            '│' ' ' '│';
            '│' ' ' '│'
        );
        const_block!(NE,
            '─' '─' '┐';
            ' ' ' ' '│';
            ' ' ' ' '│'
        );
        const_block!(E,
            ' ' ' ' '│';
            ' ' ' ' '│';
            ' ' ' ' '│'
        );
        const_block!(EE,
            '─' '─' '┐';
            ' ' ' ' '│';
            '─' '─' '┘'
        );
        const_block!(SE,
            ' ' ' ' '│';
            ' ' ' ' '│';
            '─' '─' '┘'
        );
        const_block!(S,
            ' ' ' ' ' ';
            ' ' ' ' ' ';
            '─' '─' '─'
        );
        const_block!(SS,
            '│' ' ' '│';
            '│' ' ' '│';
            '└' '─' '┘'
        );
        const_block!(SW,
            '│' ' ' ' ';
            '│' ' ' ' ';
            '└' '─' '─'
        );
        const_block!(W,
            '│' ' ' ' ';
            '│' ' ' ' ';
            '│' ' ' ' '
        );
        const_block!(WW,
            '┌' '─' '─';
            '│' ' ' ' ';
            '└' '─' '─'
        );
        const_block!(WE,
            '│' ' ' '│';
            '│' ' ' '│';
            '│' ' ' '│'
        );
        const_block!(SK,
            ' ' ' ' ' ';
            '⋮' ' ' '⋮';
            ' ' ' ' ' '
        );
    }
}
