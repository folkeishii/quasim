// https://en.wikipedia.org/wiki/Box-drawing_characters
// https://www.alt-codes.net/bullet_alt_codes.php
// https://www.compart.com/en/unicode/block/U+2B00

use std::{
    fmt::Display,
    io::{self, Write},
    iter::repeat,
};

use crossterm::style::ContentStyle;

use crate::{
    debug_terminal::show_circuit::connects::{
        Combines, ConnectEast, ConnectNorth, ConnectSouth, ConnectWest, ExtendEast, ExtendSouth,
        IsDirection, Passes,
    },
    gate::{Gate, QBits},
    instruction::Instruction,
    simulator::{DebuggableSimulator, StoredCircuitSimulator},
};

const T: bool = true;
const F: bool = false;

pub fn show_circuit<W, S>(w: &mut W, simulator: &S) -> io::Result<()>
where
    W: Write,
    S: DebuggableSimulator + StoredCircuitSimulator,
{
    let circuit = simulator.circuit();
    let mut cols = vec![Column::only_kets(repeat('0').take(circuit.n_qubits()))];
    // Add seperator
    let mut ncol = Column::only_tracks(circuit.n_qubits());
    let li = cols.len() - 1;
    cols[li].extend_east(&mut ncol);
    cols.push(ncol);

    for instruction in circuit.instructions() {
        let mut ncol = Column::from_instruction(circuit.n_qubits(), instruction);
        let li = cols.len() - 1;
        cols[li].extend_east(&mut ncol);
        cols.push(ncol);
        // Add seperator
        let mut ncol = Column::only_tracks(circuit.n_qubits());
        let li = cols.len() - 1;
        cols[li].extend_east(&mut ncol);
        cols.push(ncol);
    }

    let mut col_iters = Vec::with_capacity(cols.len());
    for col in cols.iter() {
        col_iters.push(col.into_iter());
    }
    let mut cont = true;
    while cont {
        for col in col_iters.iter_mut() {
            let Some(pp) = col.next() else {
                cont = false;
                break;
            };

            pp.queue_print(w)?;
        }
        queue_print!(w; "\r\n")?;
    }
    w.flush()
}

#[inline(always)]
/// Checks if bit 0 is 1
fn active1(bitmask: usize) -> bool {
    bitmask & 1 != 0
}
#[inline(always)]
/// Checks if bit 0 is 1
fn active_or(bitmask: usize) -> bool {
    bitmask != 0
}

type PrimitiveBlock = [[char; 3]; 3];

#[derive(Debug, Clone, Copy)]
pub enum TrackModifier {
    Ctrl,
    CtrlNot,
    Swap,
    Ket(char),
}
pub enum GateModifier {
    Ctrl,
    CtrlNot,
    Breaks,
}

#[derive(Debug, Clone, Copy)]
pub enum Primitive {
    Track {
        direction: IsDirection,
        modifier: Option<TrackModifier>,
    },
    Gate {
        walls: IsDirection,
        connected: IsDirection,
    },
}
impl Primitive {
    #[inline(always)]
    pub const fn create_track() -> Self {
        Self::create_track_with(None)
    }

    #[inline(always)]
    pub const fn create_track_with(modifier: Option<TrackModifier>) -> Self {
        Self::Track {
            direction: IsDirection {
                north: false,
                east: false,
                south: false,
                west: false,
            },
            modifier,
        }
    }

    #[inline(always)]
    pub const fn create_gate() -> Self {
        Self::Gate {
            walls: IsDirection {
                north: true,
                east: true,
                south: true,
                west: true,
            },
            connected: IsDirection {
                north: false,
                east: false,
                south: false,
                west: false,
            },
        }
    }
    //

    /// `y < 3`
    pub fn queue_print<W: Write>(&self, w: &mut W, width: usize, y: usize) -> io::Result<()> {
        let width = width + 5 - (width & 1); // ensure odd width
        let chars: String = (0..width).map(|x| self.char(width, x, y)).collect();
        queue_print!(w; chars)
    }

    /// `y < 3`
    pub fn queue_print_content<W: Write>(
        &self,
        w: &mut W,
        content: &impl Printable,
        y: usize,
    ) -> io::Result<()> {
        let width = content.length() + 5 - (content.length() & 1); // ensure odd width
        let pre_width = 2;
        let suf_width = width - content.length() - pre_width;
        let pre: String = (0..pre_width).map(|x| self.char(width, x, y)).collect();
        let suf: String = ((width - suf_width)..width)
            .map(|x| self.char(width, x, y))
            .collect();
        queue_print!(w; pre)?;
        content.queue_print(w)?;
        queue_print!(w; suf)
    }

    pub fn queue_print_shared<W: Write>(
        &self,
        w: &mut W,
        south: &Primitive,
        width: usize,
    ) -> io::Result<()> {
        let width = width + 5 - (width & 1); // ensure odd width
        let self_chars = (0..width).map(|x| self.char(width, x, 4));
        let soth_chars = (0..width).map(|x| south.char(width, x, 0));
        let chars: String = self_chars
            .zip(soth_chars)
            .map(|(slf, lwr)| slf.combined(&lwr))
            .collect();

        queue_print!(w; chars)
    }

    pub fn queue_print_shared_content<W: Write>(
        &self,
        w: &mut W,
        south: &Primitive,
        content: &impl Printable,
    ) -> io::Result<()> {
        let width = content.length() + 5 - (content.length() & 1); // ensure odd width
        let pre_width = 2;
        let suf_width = width - content.length() - pre_width;
        let self_pre = (0..pre_width).map(|x| self.char(width, x, 4));
        let self_suf = ((width - 2)..width).map(|x| self.char(width, x, 4));
        let soth_pre = (0..pre_width).map(|x| south.char(width, x, 0));
        let soth_suf = ((width - suf_width)..width).map(|x| south.char(width, x, 0));
        let pre: String = self_pre
            .zip(soth_pre)
            .map(|(slf, lwr)| slf.combined(&lwr))
            .collect();
        let suf: String = self_suf
            .zip(soth_suf)
            .map(|(slf, lwr)| slf.combined(&lwr))
            .collect();

        queue_print!(w; pre)?;
        content.queue_print(w)?;
        queue_print!(w; suf)
    }

    fn char(&self, block_width: usize, x: usize, y: usize) -> char {
        match self {
            Primitive::Track {
                direction,
                modifier,
                ..
            } => Self::track_char(direction, modifier, block_width, x, y),
            Primitive::Gate {
                walls, connected, ..
            } => Self::gate_char(walls, connected, block_width, x, y),
        }
    }

    fn track_char(
        direction: &IsDirection,
        modifier: &Option<TrackModifier>,
        block_width: usize,
        x: usize,
        y: usize,
    ) -> char {
        let ix_1_mid = Self::rep_sides_i(block_width, x);
        let ix_3_mid = Self::rep_sides_3_mid_i(block_width, x);
        match *modifier {
            Some(TrackModifier::Ctrl) => {
                Self::track_char_3_mid(direction, TrackModifier::Ctrl, ix_3_mid, y)
            }
            Some(TrackModifier::CtrlNot) => {
                Self::track_char_3_mid(direction, TrackModifier::CtrlNot, ix_3_mid, y)
            }
            Some(TrackModifier::Ket(c)) => {
                Self::track_char_3_mid(direction, TrackModifier::Ket(c), ix_3_mid, y)
            }
            Some(TrackModifier::Swap) => Self::track_char_1_mid(direction, modifier, ix_1_mid, y),
            None => Self::track_char_1_mid(direction, modifier, ix_1_mid, y),
        }
    }

    #[rustfmt::skip]
    fn track_char_1_mid(
        _direction @ &IsDirection {
            north: dn,
            east: de,
            south: ds,
            west: dw,
        }: &IsDirection,
        modifier: &Option<TrackModifier>,
        x: usize,
        y: usize,
    ) -> char {
        match (y, x, dn, de, ds, dw, *modifier) {
              (0, 1, T,  _,  _,  _,  _) => ' '.connected_south(),

              (1, 1, T,  _,  _,  _,  _) => ' '.passed_vertical(),

              (2, 0, _,  _,  _,  T,  _) => ' '.passed_horizontal(),
              (2, 1, _,  _,  _,  _,  Some(TrackModifier::Ctrl)) => '■',
              (2, 1, _,  _,  _,  _,  Some(TrackModifier::CtrlNot)) => '□',
              (2, 1, _,  _,  _,  _,  Some(TrackModifier::Ket(c))) => c,
              (2, 1, _,  _,  _,  _,  Some(TrackModifier::Swap)) => '╳',
              (2, 1, F,  F,  F,  F,  None) => ' ',
              (2, 1, F,  F,  F,  T,  None) => ' '.connected_west(),
              (2, 1, F,  F,  T,  F,  None) => ' '.connected_south(),
              (2, 1, F,  F,  T,  T,  None) => ' '.connected_west().connected_south(),
              (2, 1, F,  T,  F,  F,  None) => ' '.connected_east(),
              (2, 1, F,  T,  F,  T,  None) => ' '.connected_west().connected_east(),
              (2, 1, F,  T,  T,  F,  None) => ' '.connected_south().connected_east(),
              (2, 1, F,  T,  T,  T,  None) => ' '.connected_west().connected_south().connected_east(),
              (2, 1, T,  F,  F,  F,  None) => ' '.connected_north(),
              (2, 1, T,  F,  F,  T,  None) => ' '.connected_west().connected_north(),
              (2, 1, T,  F,  T,  F,  None) => ' '.connected_south().connected_north(),
              (2, 1, T,  F,  T,  T,  None) => ' '.connected_west().connected_south().connected_north(),
              (2, 1, T,  T,  F,  F,  None) => ' '.connected_east().connected_north(),
              (2, 1, T,  T,  F,  T,  None) => ' '.connected_west().connected_east().connected_north(),
              (2, 1, T,  T,  T,  F,  None) => ' '.connected_south().connected_east().connected_north(),
              (2, 1, T,  T,  T,  T,  None) => ' '.connected_west().connected_south().connected_east().connected_north(),
              (2, 2, _,  T,  _,  _,  _) => ' '.passed_horizontal(),

              (3, 1, _,  _,  T,  _,  _) => ' '.passed_vertical(),

              (4, 1, _,  _,  T,  _,  _) => ' '.connected_north(),

              _ => ' ',
        }
    }

    #[rustfmt::skip]
    fn track_char_3_mid(
        _direction @ &IsDirection {
            north: dn,
            east: de,
            south: ds,
            west: dw,
        }: &IsDirection,
        modifier: TrackModifier,
        x: usize,
        y: usize,
    ) -> char {
        match (y, x, dn, de, ds, dw, modifier) {
              (0, 2, T,  _,  _,  _, _) => ' '.connected_south(),

              (1, 0, _,  _,  _,  _, _) => ' ',
              (1, 1, _,  _,  _,  _, _) => ' ',
              (1, 2, T,  _,  _,  _, _) => ' '.passed_horizontal().connected_north(),
              (1, 3, _,  _,  _,  _, _) => ' ',
              (1, 4, _,  _,  _,  _, _) => ' ',

              (2, 0, _,  _,  _,  _, TrackModifier::Ket(_)) => ' ',
              (2, 0, _,  _,  _,  F, _) => ' ',
              (2, 0, _,  _,  _,  T, _) => ' '.passed_horizontal(),
              (2, 1, _,  _,  _,  _, TrackModifier::Ket(_)) => '|',
              (2, 1, _,  _,  _,  T, _) => ' '.passed_vertical().connected_west(),
              (2, 2, _,  _,  _,  _, TrackModifier::Ctrl) => '■',
              (2, 2, _,  _,  _,  _, TrackModifier::CtrlNot) => '□',
              (2, 2, _,  _,  _,  _, TrackModifier::Ket(c)) => c,
              (2, 2, _,  _,  _,  _, TrackModifier::Swap) => '╳',
              (2, 3, _,  _,  _,  _, TrackModifier::Ket(_)) => '⟩',
              (2, 3, _,  T,  _,  _, _) => ' '.passed_vertical().connected_east(),
              (2, 4, _,  _,  _,  _, TrackModifier::Ket(_)) => ' ',
              (2, 4, _,  F,  _,  _, _) => ' ',
              (2, 4, _,  T,  _,  _, _) => ' '.passed_horizontal(),

              (3, 0, _,  _,  _,  _, _) => ' ',
              (3, 1, _,  _,  _,  _, _) => ' ',
              (3, 2, _,  _,  T,  _, _) => ' '.passed_horizontal().connected_south(),
              (3, 3, _,  _,  _,  _, _) => ' ',
              (3, 4, _,  _,  _,  _, _) => ' ',

              (4, 2, _,  _,  T,  _, _) => ' '.connected_north(),

              _ => ' '
        }
    }

    #[rustfmt::skip]
    fn gate_char(
        _walls @ &IsDirection {
            north: wn,
            east: we,
            south: ws,
            west: ww,
        }: &IsDirection,
        _connected @ &IsDirection {
            north: cn,
            east: ce,
            south: cs,
            west: cw,
            ..
        }: &IsDirection,
        block_width: usize,
        x: usize,
        y: usize,
    ) -> char {
        let iy = y;
        let ix = Self::gate_ix(block_width, x);

        match (iy, ix, wn, we, ws, ww, cn, ce, cs, cw) {
              (0,  0,  F,  _,  _,  T,  _,  _,  _,  _) => ' '.connected_south(),
              (0,  2,  _,  _,  _,  _,  T,  _,  _,  _) => ' '.connected_south(),
              (0,  4,  F,  T,  _,  _,  _,  _,  _,  _) => ' '.connected_south(),

              (1,  0,  F,  _,  _,  T,  _,  _,  _,  _) => ' '.passed_vertical(),
              (1,  0,  T,  _,  _,  F,  _,  _,  _,  _) => ' '.passed_horizontal(),
              (1,  0,  T,  _,  _,  T,  _,  _,  _,  _) => ' '.connected_east().connected_south(),

              (1,  1,  T,  _,  _,  _,  _,  _,  _,  _) => ' '.passed_horizontal(),

              (1,  2,  T,  _,  _,  _,  F,  _,  _,  _) => ' '.passed_horizontal(),
              (1,  2,  T,  _,  _,  _,  T,  _,  _,  _) => ' '.passed_horizontal().connected_north(),

              (1,  3,  T,  _,  _,  _,  _,  _,  _,  _) => ' '.passed_horizontal(),

              (1,  4,  F,  T,  _,  _,  _,  _,  _,  _) => ' '.passed_vertical(),
              (1,  4,  T,  F,  _,  _,  _,  _,  _,  _) => ' '.passed_horizontal(),
              (1,  4,  T,  T,  _,  _,  _,  _,  _,  _) => ' '.connected_west().connected_south(),

              (2,  0,  _,  _,  _,  T,  _,  _,  _,  F) => ' '.passed_vertical(),
              (2,  0,  _,  _,  _,  T,  _,  _,  _,  T) => ' '.passed_vertical().connected_west(),

              (2,  4,  _,  T,  _,  _,  _,  F,  _,  _) => ' '.passed_vertical(),
              (2,  4,  _,  T,  _,  _,  _,  T,  _,  _) => ' '.passed_vertical().connected_east(),

              (3,  0,  _,  _,  F,  T,  _,  _,  _,  _) => ' '.passed_vertical(),
              (3,  0,  _,  _,  T,  F,  _,  _,  _,  _) => ' '.passed_horizontal(),
              (3,  0,  _,  _,  T,  T,  _,  _,  _,  _) => ' '.connected_east().connected_north(),

              (3,  1,  _,  _,  T,  _,  _,  _,  _,  _) => ' '.passed_horizontal(),

              (3,  2,  _,  _,  T,  _,  _,  _,  F,  _) => ' '.passed_horizontal(),
              (3,  2,  _,  _,  T,  _,  _,  _,  T,  _) => ' '.passed_horizontal().connected_south(),

              (3,  3,  _,  _,  T,  _,  _,  _,  _,  _) => ' '.passed_horizontal(),

              (3,  4,  _,  T,  F,  _,  _,  _,  _,  _) => ' '.passed_vertical(),
              (3,  4,  _,  F,  T,  _,  _,  _,  _,  _) => ' '.passed_horizontal(),
              (3,  4,  _,  T,  T,  _,  _,  _,  _,  _) => ' '.connected_west().connected_north(),

              (4,  0,  _,  _,  F,  T,  _,  _,  _,  _) => ' '.connected_north(),
              (4,  2,  _,  _,  _,  _,  _,  _,  T,  _) => ' '.connected_north(),
              (4,  4,  _,  T,  F,  _,  _,  _,  _,  _) => ' '.connected_north(),

              _ => ' ',
        }
    }

    const fn gate_ix(block_width: usize, x: usize) -> usize {
        Self::rep_mid_i(block_width, x)
    }

    #[inline(always)]
    const fn rep_mid_i(block_width: usize, i: usize) -> usize {
        // 1: i *= 2
        // 2: i -= bw
        // 3: let is = i.signum()
        //    i /= bw
        // 5: i += is
        // 4: i += 2
        // odd block_width = 7 ==> bw = 6
        // index:
        //    0   0  -6  -1  -2   0
        //    1   2  -4   0  -1   1
        //    2   4  -2   0  -1   1
        //    3   6   0   0   0   2
        //    4   8   2   0   1   3
        //    5  10   4   0   1   3
        //    6  12   6   1   2   4
        //
        let bw = (block_width - 1) as isize;
        let i = 2 * i as isize - bw;
        let is = i.signum();
        (i / bw + is + 2) as usize
    }

    #[inline(always)]
    const fn rep_sides_i(block_width: usize, i: usize) -> usize {
        // 1: i *= 2
        // 2: i -= bw
        // 3: i = i.signum()
        // 4: i += 1
        // odd block_width = 5 ==> bw = 4
        // index:
        //    0   0  -4  -1   0
        //    1   2  -2  -1   0
        //    2   4   0   0   1
        //    3   6   2   1   2
        //    4   8   4   1   2
        // even block_width = 6 ==> bw = 5
        // index:
        //    0   0  -5  -1   0
        //    1   2  -3  -1   0
        //    2   4  -1  -1   0
        //    3   6   1   1   2
        //    4   8   3   1   2
        //    5  10   5   1   2
        //
        let bw = (block_width - 1) as isize;
        let i = i as isize;
        ((2 * i - bw).signum() + 1) as usize
    }

    #[inline(always)]
    // returned value is within range [0,5)
    const fn rep_sides_3_mid_i(block_width: usize, i: usize) -> usize {
        // 1: i *= 2
        // 2: i -= bw
        // 3: let is = i.signum()
        //    i -= 2 * is
        // 4: i = i.signum()
        // 5: i += is
        // 4: i += 2
        // odd block_width = 7 ==> bw = 6
        // index:
        //    0   0  -6  -4  -1  -2   0
        //    1   2  -4  -2  -1  -2   0
        //    2   4  -2   0   0  -1   1
        //    3   6   0   0   0   0   2
        //    4   8   2   0   0   1   3
        //    5  10   4   2   1   2   4
        //    6  12   6   4   1   2   4
        let bw = (block_width - 1) as isize;
        let i = 2 * i as isize - bw;
        let is = i.signum();
        ((i - 2 * is).signum() + is + 2) as usize
    }
}
impl ConnectNorth for Primitive {
    fn connect_north(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.north = true,
            Primitive::Gate { connected, .. } => connected.north = true,
        }
    }

    fn clear_north(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.north = false,
            Primitive::Gate { connected, .. } => connected.north = false,
        }
    }
}
impl ConnectEast for Primitive {
    fn connect_east(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.east = true,
            Primitive::Gate { connected, .. } => connected.east = true,
        }
    }

    fn clear_east(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.east = false,
            Primitive::Gate { connected, .. } => connected.east = false,
        }
    }
}
impl ConnectSouth for Primitive {
    fn connect_south(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.south = true,
            Primitive::Gate { connected, .. } => connected.south = true,
        }
    }

    fn clear_south(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.south = false,
            Primitive::Gate { connected, .. } => connected.south = false,
        }
    }
}
impl ConnectWest for Primitive {
    fn connect_west(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.west = true,
            Primitive::Gate { connected, .. } => connected.west = true,
        }
    }

    fn clear_west(&mut self) {
        match self {
            Primitive::Track { direction, .. } => direction.west = false,
            Primitive::Gate { connected, .. } => connected.west = false,
        }
    }
}
impl ExtendSouth for Primitive {
    fn extend_south(&mut self, sth: &mut Self) {
        match (self, sth) {
            (
                Primitive::Gate {
                    walls: IsDirection { south: north, .. },
                    ..
                },
                Primitive::Gate {
                    walls: IsDirection { north: south, .. },
                    ..
                },
            ) => {
                *north = false;
                *south = false;
            }
            (north, south) => {
                north.connect_south();
                south.connect_north();
            }
        }
    }
}
impl ExtendEast for Primitive {
    fn extend_east(&mut self, est: &mut Self) {
        match (self, est) {
            (west, east) => {
                west.connect_east();
                east.connect_west();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    primitives: PrimitiveVec,
    groups: CombinedRanges,
    gate_content: EitherContent,
}
impl Column {
    pub fn from_instruction(nqubits: usize, instruction: &Instruction) -> Self {
        match instruction {
            Instruction::Gate(gate) => Self::from_gate(nqubits, gate),
            Instruction::Measurement(qbits, _) => {
                let mut qbits = qbits.get_bitstring();
                let mut column = if qbits & 1 == 1 {
                    Column::init_with_gate(String::from("╭─╱─╮"))
                } else {
                    Column::init_with_track(String::from("╭─╱─╮"))
                };
                for _ in 1..nqubits {
                    qbits >>= 1;
                    if qbits & 1 == 1 {
                        column.close_with_gate();
                    } else {
                        column.close_with_track();
                    }
                }
                column
            }
            Instruction::Jump(_) => todo!(),
            Instruction::JumpIf(_, _) => todo!(),
            Instruction::Assign(_, _) => todo!(),
        }
    }

    pub fn from_gate(nqubits: usize, gate: &Gate) -> Self {
        match gate.get_type() {
            crate::gate::GateType::X => Self::from_common_gate(
                nqubits,
                String::from("X"),
                gate.get_target_bits(),
                gate.get_control_bits(),
            ),
            crate::gate::GateType::Y => Self::from_common_gate(
                nqubits,
                String::from("Y"),
                gate.get_target_bits(),
                gate.get_control_bits(),
            ),
            crate::gate::GateType::Z => Self::from_common_gate(
                nqubits,
                String::from("Z"),
                gate.get_target_bits(),
                gate.get_control_bits(),
            ),
            crate::gate::GateType::H => Self::from_common_gate(
                nqubits,
                String::from("H"),
                gate.get_target_bits(),
                gate.get_control_bits(),
            ),
            crate::gate::GateType::SWAP => {
                Self::from_swap_gate(nqubits, gate.get_target_bits(), gate.get_control_bits())
            }
            crate::gate::GateType::S => Self::from_common_gate(
                nqubits,
                String::from("S"),
                gate.get_target_bits(),
                gate.get_control_bits(),
            ),
            crate::gate::GateType::U(_, _, _) => Self::from_common_gate(
                nqubits,
                String::from("U"),
                gate.get_target_bits(),
                gate.get_control_bits(),
            ),
        }
    }

    pub fn from_common_gate<I: Into<EitherContent>>(
        nqubits: usize,
        content: I,
        targets: QBits,
        controls: QBits,
    ) -> Self {
        let mut targets_north = 0usize;
        let mut target = active1(targets.get_bitstring());
        let mut targets_south = targets.get_bitstring() >> 1;
        let mut controls_north = 0usize;
        let mut control = active1(controls.get_bitstring());
        let mut controls_south = controls.get_bitstring() >> 1;
        let mut column;
        if target {
            column = Column::init_with_gate(content);
        } else if control {
            column = Column::init_with_track_modifier(content, Some(TrackModifier::Ctrl));
        } else {
            column = Column::init_with_track(content);
        }

        for _ in 1..nqubits {
            targets_north <<= 1;
            targets_north |= target as usize;
            target = active1(targets_south);
            targets_south >>= 1;
            controls_north <<= 1;
            controls_north |= control as usize;
            control = active1(controls_south);
            controls_south >>= 1;

            if target {
                if active1(targets_north) || active_or(controls_north) {
                    column.extend_with_gate();
                } else {
                    column.close_with_gate();
                }
            } else if control {
                if active_or(targets_north)
                    || (active_or(controls_north) && active_or(targets_south))
                {
                    column.extend_with_track_modifier(Some(TrackModifier::Ctrl));
                } else {
                    column.close_with_track_modifier(Some(TrackModifier::Ctrl));
                }
            } else {
                if (active_or(targets_north) && active_or(controls_south))
                    || (active_or(targets_south) && active_or(controls_north))
                {
                    column.extend_with_track();
                } else {
                    column.close_with_track();
                }
            }
        }

        column
    }

    pub fn from_swap_gate(nqubits: usize, targets: QBits, controls: QBits) -> Self {
        let mut targets_north = 0usize;
        let mut target = active1(targets.get_bitstring());
        let mut targets_south = targets.get_bitstring() >> 1;
        let mut controls_north = 0usize;
        let mut control = active1(controls.get_bitstring());
        let mut controls_south = controls.get_bitstring() >> 1;
        let mut column;
        if target {
            column = Column::init_with_track_modifier(1, Some(TrackModifier::Swap));
        } else if control {
            column = Column::init_with_track_modifier(1, Some(TrackModifier::Ctrl));
        } else {
            column = Column::init_with_track(1);
        }

        for _ in 1..nqubits {
            targets_north <<= 1;
            targets_north |= target as usize;
            target = active1(targets_south);
            targets_south >>= 1;
            controls_north <<= 1;
            controls_north |= control as usize;
            control = active1(controls_south);
            controls_south >>= 1;

            if target {
                if active_or(targets_north) || active_or(controls_north) {
                    column.extend_with_track_modifier(Some(TrackModifier::Swap));
                } else {
                    column.close_with_track_modifier(Some(TrackModifier::Swap));
                }
            } else if control {
                if active_or(targets_north)
                    || (active_or(controls_north) && active_or(targets_south))
                {
                    column.extend_with_track_modifier(Some(TrackModifier::Ctrl));
                } else {
                    column.close_with_track_modifier(Some(TrackModifier::Ctrl));
                }
            } else {
                if (active_or(targets_north) && active_or(controls_south))
                    || (active_or(targets_south) && active_or(controls_north))
                    || (active_or(targets_north) && active_or(targets_south))
                {
                    column.extend_with_track();
                } else {
                    column.close_with_track();
                }
            }
        }

        column
    }

    #[allow(dead_code, unreachable_code, unused_variables)]
    /// Have not found a suitable unicode for cnot
    pub fn from_cnot_gate(nqubits: usize, targets: QBits, controls: QBits) -> Self {
        todo!();
        let mut targets_north = 0usize;
        let mut target = active1(targets.get_bitstring());
        let mut targets_south = targets.get_bitstring() >> 1;
        let mut controls_north = 0usize;
        let mut control = active1(controls.get_bitstring());
        let mut controls_south = controls.get_bitstring() >> 1;
        let mut column;
        if target {
            column = Column::init_with_gate(1);
        } else if control {
            column = Column::init_with_track(1);
        } else {
            column = Column::init_with_track(1);
        }

        for _ in 1..nqubits {
            targets_north <<= 1;
            targets_north |= target as usize;
            target = active1(targets_south);
            targets_south >>= 1;
            controls_north <<= 1;
            controls_north |= control as usize;
            control = active1(controls_south);
            controls_south >>= 1;

            if target {
                if active1(targets_north) || active_or(controls_north) {
                    // column.extend_with_track_modifier(Some(TrackModifier::CNOT));
                } else {
                    // column.close_with_track_modifier(Some(TrackModifier::CNOT));
                }
            } else if control {
                if active_or(targets_north)
                    || (active_or(controls_north) && active_or(targets_south))
                {
                    column.extend_with_track_modifier(Some(TrackModifier::Ctrl));
                } else {
                    column.close_with_track_modifier(Some(TrackModifier::Ctrl));
                }
            } else {
                if (active_or(targets_north) && active_or(controls_south))
                    || (active_or(targets_south) && active_or(controls_north))
                {
                    column.extend_with_track();
                } else {
                    column.close_with_track();
                }
            }
        }

        column
    }

    pub fn only_tracks(nqubits: usize) -> Self {
        let mut column = Self::init_with_track(1);
        for _ in 1..nqubits {
            column.close_with_track();
        }
        column
    }

    pub fn only_kets<I: IntoIterator<Item = char>>(ket_ids: I) -> Self {
        let mut kets = ket_ids.into_iter();
        let mut column = kets
            .next()
            .map(|c| Self::init_with_track_modifier(1, Some(TrackModifier::Ket(c))))
            .expect("Expected at least one ket");

        while let Some(c) = kets.next() {
            column.close_with_track_modifier(Some(TrackModifier::Ket(c)));
        }

        column
    }

    pub fn init_with_track<I: Into<EitherContent>>(content: I) -> Self {
        Self::init_with_track_modifier(content, None)
    }

    pub fn init_with_track_modifier<I: Into<EitherContent>>(
        content: I,
        modifier: Option<TrackModifier>,
    ) -> Self {
        Self {
            primitives: PrimitiveVec::init_with_track_modifier(modifier),
            groups: OtherRange::new().into(),
            gate_content: content.into(),
        }
    }

    pub fn init_with_gate<I: Into<EitherContent>>(content: I) -> Self {
        Self {
            primitives: PrimitiveVec::init_with_gate(),
            groups: GateRange::new().into(),
            gate_content: content.into(),
        }
    }

    pub fn extend_with_gate(&mut self) {
        match self.primitives.last_mut() {
            north @ Primitive::Track { .. } => {
                let mut primitive = Primitive::create_gate();
                north.extend_south(&mut primitive);
                self.primitives.0.push(primitive);
                self.groups.close_with_gate();
            }
            north @ Primitive::Gate { .. } => {
                let mut primitive = Primitive::create_gate();
                north.extend_south(&mut primitive);
                self.primitives.0.push(primitive);
                self.groups.extend_once();
            }
        }
    }

    pub fn extend_with_track(&mut self) {
        Self::extend_with_track_modifier(self, None);
    }

    pub fn extend_with_track_modifier(&mut self, modifier: Option<TrackModifier>) {
        match self.primitives.last_mut() {
            north @ Primitive::Track { .. } => {
                let mut primitive = Primitive::create_track_with(modifier);
                north.extend_south(&mut primitive);
                self.primitives.0.push(primitive);
                self.groups.extend_once();
            }
            north @ Primitive::Gate { .. } => {
                let mut primitive = Primitive::create_track_with(modifier);
                north.extend_south(&mut primitive);
                self.primitives.0.push(primitive);
                self.groups.close_with_other();
            }
        }
    }

    pub fn close_with_gate(&mut self) {
        self.primitives.0.push(Primitive::create_gate());
        self.groups.close_with_gate();
    }

    pub fn close_with_track(&mut self) {
        Self::close_with_track_modifier(self, None);
    }

    pub fn close_with_track_modifier(&mut self, modifier: Option<TrackModifier>) {
        self.primitives
            .0
            .push(Primitive::create_track_with(modifier));
        self.groups.close_with_other();
    }

    #[inline(always)]
    fn first(&self) -> &Primitive {
        self.primitives.first()
    }

    #[inline(always)]
    fn first_mut(&mut self) -> &mut Primitive {
        self.primitives.first_mut()
    }

    #[inline(always)]
    fn last(&self) -> &Primitive {
        self.primitives.last()
    }

    #[inline(always)]
    fn last_mut(&mut self) -> &mut Primitive {
        self.primitives.last_mut()
    }
}
impl ExtendEast for Column {
    fn extend_east(&mut self, est: &mut Self) {
        for (wst, est) in self
            .primitives
            .0
            .iter_mut()
            .zip(est.primitives.0.iter_mut())
        {
            wst.extend_east(est);
        }
    }
}
impl<'a> IntoIterator for &'a Column {
    type Item = PrimitiveWithContent<'a>;
    type IntoIter = ColumnIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ColumnIterator {
            primitives: &self.primitives,
            groups: self.groups.into_iter(),
            gate_content: self.gate_content.as_ref(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColumnIterator<'a> {
    primitives: &'a PrimitiveVec,
    groups: CombinedIndicesIterator<'a>,
    gate_content: EitherContentRef<'a>,
}
impl<'a> Iterator for ColumnIterator<'a> {
    type Item = PrimitiveWithContent<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(index) = self.groups.next() else {
            return None;
        };

        let (index, content) = match index {
            EitherIndex::Gate(GateIndex::Named(index)) => (index, self.gate_content),
            EitherIndex::Gate(GateIndex::Normal(index)) | EitherIndex::Other(index) => {
                (index, self.gate_content.as_width())
            }
        };

        let Some(primitive) = self.primitives.get(index) else {
            return None;
        };

        Some(primitive.with(content))
    }
}

#[derive(Debug, Clone)]
pub struct PrimitiveVec(Vec<Primitive>);
impl PrimitiveVec {
    pub fn init_with_track() -> Self {
        Self::init_with_track_modifier(None)
    }

    pub fn init_with_track_modifier(modifier: Option<TrackModifier>) -> Self {
        Self(vec![Primitive::create_track_with(modifier)])
    }

    pub fn init_with_gate() -> Self {
        Self(vec![Primitive::create_gate()])
    }

    pub fn get(&'_ self, index: SharedRelIndex) -> Option<SharedPrimitive<'_>> {
        match index {
            SharedRelIndex::Single { north, rel_y } => {
                self.0.get(north).map(|p| SharedPrimitive::Single {
                    north: p,
                    rel_y: rel_y,
                })
            }
            SharedRelIndex::Shared { north, south } => self
                .0
                .get(north)
                .zip(self.0.get(south))
                .map(|(n, s)| SharedPrimitive::Shared { north: n, south: s }),
        }
    }

    #[inline(always)]
    fn first(&self) -> &Primitive {
        self.0.first().expect("Broken invariant")
    }

    #[inline(always)]
    fn first_mut(&mut self) -> &mut Primitive {
        self.0.first_mut().expect("Broken invariant")
    }

    #[inline(always)]
    fn last(&self) -> &Primitive {
        self.0.last().expect("Broken invariant")
    }

    #[inline(always)]
    fn last_mut(&mut self) -> &mut Primitive {
        self.0.last_mut().expect("Broken invariant")
    }
}

#[derive(Debug, Clone)]
pub struct PrimitiveWithContent<'a> {
    primitive: SharedPrimitive<'a>,
    content: EitherContentRef<'a>,
}
impl<'a> PrimitiveWithContent<'a> {
    pub fn queue_print<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match &self.content {
            EitherContentRef::Styled(content) => self.primitive.queue_print_content(w, content),
            EitherContentRef::Raw(content) => self.primitive.queue_print_content(w, content),
            EitherContentRef::Width(width) => self.primitive.queue_print(w, *width),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CombinedRanges(Vec<EitherRange>);
impl CombinedRanges {
    pub fn extend_once(&mut self) {
        self.unchecked_last_mut().extend_once();
    }

    pub fn close_with_gate(&mut self) {
        let range = self.unchecked_last_mut().close_with_gate();
        self.0.push(range.into());
    }

    pub fn close_with_other(&mut self) {
        let range = self.unchecked_last_mut().close_with_other();
        self.0.push(range.into());
    }

    pub fn end_inclusive(&self) -> SharedRelIndex {
        self.unchecked_last().end_inclusive()
    }

    pub fn end_non_inclusive(&self) -> SharedRelIndex {
        self.unchecked_last().end_non_inclusive()
    }

    #[inline(always)]
    fn last(&self) -> &EitherRange {
        self.unchecked_last()
    }

    #[inline(always)]
    fn unchecked_last(&self) -> &EitherRange {
        &self.0[self.0.len() - 1]
    }

    #[inline(always)]
    fn unchecked_last_mut(&mut self) -> &mut EitherRange {
        let i = self.0.len() - 1;
        &mut self.0[i]
    }
}
impl From<GateRange> for CombinedRanges {
    fn from(value: GateRange) -> Self {
        Self(vec![value.into()])
    }
}
impl From<OtherRange> for CombinedRanges {
    fn from(value: OtherRange) -> Self {
        Self(vec![value.into()])
    }
}
impl From<EitherRange> for CombinedRanges {
    fn from(value: EitherRange) -> Self {
        Self(vec![value])
    }
}
impl<'a> IntoIterator for &'a CombinedRanges {
    type Item = EitherIndex;
    type IntoIter = CombinedIndicesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CombinedIndicesIterator {
            next: &self.0[1..],
            current: self.0[0].into_iter(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CombinedIndicesIterator<'a> {
    next: &'a [EitherRange],
    current: EitherIndexIterator,
}
impl<'a> Iterator for CombinedIndicesIterator<'a> {
    type Item = EitherIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.current.next() {
            return Some(item);
        }

        if let Some(range) = self.next.get(0) {
            self.next = &self.next[1..];
            self.current = range.into_iter();
            self.current.next()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EitherRange {
    Gate(GateRange),
    Other(OtherRange),
}
impl EitherRange {
    pub fn extend_once(&mut self) {
        match self {
            EitherRange::Gate(gate_range) => gate_range.extend_once(),
            EitherRange::Other(other_range) => other_range.extend_once(),
        }
    }

    pub fn close_with_gate(&mut self) -> GateRange {
        let gate = match self {
            EitherRange::Gate(gate_range) => gate_range.close_with_gate(),
            EitherRange::Other(other_range) => other_range.close_with_gate(),
        };
        gate
    }

    pub fn close_with_other(&mut self) -> OtherRange {
        let other = match self {
            EitherRange::Gate(gate_range) => gate_range.close_with_other(),
            EitherRange::Other(other_range) => other_range.close_with_other(),
        };
        other
    }

    pub fn end_inclusive(&self) -> SharedRelIndex {
        match self {
            EitherRange::Gate(gate_range) => gate_range.end_inclusive(),
            EitherRange::Other(other_range) => other_range.end_inclusive(),
        }
    }

    pub fn end_non_inclusive(&self) -> SharedRelIndex {
        let mut index = self.end_inclusive();
        index.next();
        index
    }
}
impl From<GateRange> for EitherRange {
    fn from(value: GateRange) -> Self {
        Self::Gate(value)
    }
}
impl From<OtherRange> for EitherRange {
    fn from(value: OtherRange) -> Self {
        Self::Other(value)
    }
}
impl IntoIterator for EitherRange {
    type Item = EitherIndex;
    type IntoIter = EitherIndexIterator;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            EitherRange::Gate(gate_range) => EitherIndexIterator::Gate(gate_range.into_iter()),
            EitherRange::Other(other_range) => EitherIndexIterator::Other(other_range.into_iter()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EitherIndexIterator {
    Gate(GateIndexIterator),
    Other(OtherIndexIterator),
}
impl Iterator for EitherIndexIterator {
    type Item = EitherIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EitherIndexIterator::Gate(g) => g.next().map(|i| EitherIndex::Gate(i)),
            EitherIndexIterator::Other(o) => o.next().map(|i| EitherIndex::Other(i)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EitherIndex {
    Gate(GateIndex),
    Other(SharedRelIndex),
}

#[derive(Debug, Clone, Copy)]
pub struct OtherRange {
    start: SharedRelIndex,
    primitive_count: usize,
    closed: bool,
}
impl OtherRange {
    pub fn new() -> Self {
        OtherRange {
            start: SharedRelIndex::Single { north: 0, rel_y: 0 },
            primitive_count: 1,
            closed: true,
        }
    }

    pub fn extend_once(&mut self) {
        self.primitive_count += 1;
    }

    pub fn close_with_gate(&mut self) -> GateRange {
        self.closed = false;
        let mut ni = self.end_non_inclusive();
        ni.next();
        ni.next();
        GateRange {
            start: self.end_non_inclusive(),
            primitive_count: 1,
            named_index: ni,
            closed: true,
        }
    }

    pub fn close_with_other(&mut self) -> OtherRange {
        self.closed = false;
        OtherRange {
            start: self.end_non_inclusive(),
            primitive_count: 1,
            closed: true,
        }
    }

    pub fn end_inclusive(&self) -> SharedRelIndex {
        if self.closed {
            match self.start {
                SharedRelIndex::Single { north, .. } => SharedRelIndex::Single {
                    north: north + self.primitive_count - 1,
                    rel_y: 4,
                },
                SharedRelIndex::Shared { south, .. } => SharedRelIndex::Single {
                    north: south + self.primitive_count - 1,
                    rel_y: 4,
                },
            }
        } else {
            match self.start {
                SharedRelIndex::Single { north, .. } => SharedRelIndex::Single {
                    north: north + self.primitive_count - 1,
                    rel_y: 3,
                },
                SharedRelIndex::Shared { south, .. } => SharedRelIndex::Single {
                    north: south + self.primitive_count - 1,
                    rel_y: 3,
                },
            }
        }
    }

    pub fn end_non_inclusive(&self) -> SharedRelIndex {
        let mut index = self.end_inclusive();
        index.next();
        index
    }
}
impl IntoIterator for OtherRange {
    type Item = SharedRelIndex;
    type IntoIter = OtherIndexIterator;

    fn into_iter(self) -> Self::IntoIter {
        OtherIndexIterator {
            current: self.start,
            end_inclusive: self.end_inclusive(),
            exhausted: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OtherIndexIterator {
    current: SharedRelIndex,
    end_inclusive: SharedRelIndex,
    exhausted: bool,
}
impl Iterator for OtherIndexIterator {
    type Item = SharedRelIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let item = self.current;

        if self.current.north() == self.end_inclusive.north()
            && self.current.to_rel_y() == self.end_inclusive.to_rel_y()
        {
            self.exhausted = true;
            return Some(self.end_inclusive);
        }

        self.current.next();

        Some(item)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GateRange {
    start: SharedRelIndex,
    primitive_count: usize,
    named_index: SharedRelIndex,
    closed: bool,
}
impl GateRange {
    pub fn new() -> Self {
        Self {
            start: SharedRelIndex::Single { north: 0, rel_y: 0 },
            named_index: SharedRelIndex::Single { north: 0, rel_y: 2 },
            primitive_count: 1,
            closed: true,
        }
    }

    pub fn extend_once(&mut self) {
        self.primitive_count += 1;
        self.named_index = match self.named_index {
            SharedRelIndex::Single { north, .. } => SharedRelIndex::Shared {
                north,
                south: north + 1,
            },
            SharedRelIndex::Shared { south, .. } => SharedRelIndex::Single {
                north: south,
                rel_y: 2,
            },
        };
    }

    pub fn close_with_gate(&mut self) -> GateRange {
        self.closed = false;
        let mut ni = self.end_non_inclusive();
        ni.next();
        ni.next();
        GateRange {
            start: self.end_non_inclusive(),
            primitive_count: 1,
            named_index: ni,
            closed: true,
        }
    }

    pub fn close_with_other(&mut self) -> OtherRange {
        self.closed = false;
        OtherRange {
            start: self.end_non_inclusive(),
            primitive_count: 1,
            closed: true,
        }
    }

    pub fn end_inclusive(&self) -> SharedRelIndex {
        if self.closed {
            match self.start {
                SharedRelIndex::Single { north, .. } => SharedRelIndex::Single {
                    north: north + self.primitive_count - 1,
                    rel_y: 4,
                },
                SharedRelIndex::Shared { south, .. } => SharedRelIndex::Single {
                    north: south + self.primitive_count - 1,
                    rel_y: 4,
                },
            }
        } else {
            match self.start {
                SharedRelIndex::Single { north, .. } => SharedRelIndex::Single {
                    north: north + self.primitive_count - 1,
                    rel_y: 3,
                },
                SharedRelIndex::Shared { south, .. } => SharedRelIndex::Single {
                    north: south + self.primitive_count - 1,
                    rel_y: 3,
                },
            }
        }
    }

    pub fn end_non_inclusive(&self) -> SharedRelIndex {
        let mut index = self.end_inclusive();
        index.next();
        index
    }
}
impl IntoIterator for GateRange {
    type Item = GateIndex;
    type IntoIter = GateIndexIterator;

    fn into_iter(self) -> Self::IntoIter {
        GateIndexIterator {
            current: self.start,
            end_inclusive: self.end_inclusive(),
            named_index: self.named_index,
            exhausted: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateIndexIterator {
    current: SharedRelIndex,
    end_inclusive: SharedRelIndex,
    named_index: SharedRelIndex,
    exhausted: bool,
}
impl Iterator for GateIndexIterator {
    type Item = GateIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let item = match (
            self.current == self.named_index,
            self.current.north() == self.end_inclusive.north()
                && self.current.to_rel_y() == self.end_inclusive.to_rel_y(),
        ) {
            //(include name , reached end)
            (true, true) => {
                self.exhausted = true;
                GateIndex::Named(self.end_inclusive)
            }
            (true, false) => GateIndex::Named(self.current),
            (false, true) => {
                self.exhausted = true;
                GateIndex::Normal(self.end_inclusive)
            }
            (false, false) => GateIndex::Normal(self.current),
        };
        self.current.next();

        Some(item)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateIndex {
    Normal(SharedRelIndex),
    Named(SharedRelIndex),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// `assert!(rel_y < 3)`
/// `assert_eq!(south - north, 1)`
pub enum SharedRelIndex {
    Single { north: usize, rel_y: usize },
    Shared { north: usize, south: usize },
}
impl SharedRelIndex {
    #[inline(always)]
    pub const fn north(&self) -> usize {
        match self {
            SharedRelIndex::Single { north, .. } | SharedRelIndex::Shared { north, .. } => *north,
        }
    }

    #[inline(always)]
    /// Returns `Ok(usize)` containing `rel_y` if `matches!(self, Self::Single {..})`
    ///
    /// Returns `Err(usize)` containing `south` if `matches!(self, Self::Shared {..})`
    const fn rel_y(&self) -> Result<usize, usize> {
        match self {
            SharedRelIndex::Single { rel_y, .. } => Ok(*rel_y),
            SharedRelIndex::Shared { south, .. } => Err(*south),
        }
    }

    #[inline(always)]
    const fn to_rel_y(&self) -> usize {
        match self {
            SharedRelIndex::Single { rel_y, .. } => *rel_y,
            SharedRelIndex::Shared { .. } => 4,
        }
    }
}
impl Iterator for SharedRelIndex {
    type Item = Self;

    fn next(&mut self) -> Option<Self::Item> {
        let item = *self;
        match self {
            SharedRelIndex::Single { north, rel_y: 3 } => {
                *self = SharedRelIndex::Shared {
                    north: *north,
                    south: *north + 1,
                }
            }
            SharedRelIndex::Single { rel_y, .. } => {
                *rel_y += 1;
            }
            SharedRelIndex::Shared { south, .. } => {
                *self = SharedRelIndex::Single {
                    north: *south,
                    rel_y: 1,
                }
            }
        };
        Some(item)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SharedPrimitive<'a> {
    Single {
        north: &'a Primitive,
        rel_y: usize,
    },
    Shared {
        north: &'a Primitive,
        south: &'a Primitive,
    },
}
impl<'a> SharedPrimitive<'a> {
    pub fn queue_print<W: Write>(&self, w: &mut W, width: usize) -> io::Result<()> {
        match self {
            SharedPrimitive::Single { north, rel_y } => north.queue_print(w, width, *rel_y),
            SharedPrimitive::Shared { north, south } => north.queue_print_shared(w, south, width),
        }
    }

    pub fn queue_print_content<W: Write>(
        &self,
        w: &mut W,
        content: &impl Printable,
    ) -> io::Result<()> {
        match self {
            SharedPrimitive::Single { north, rel_y } => {
                north.queue_print_content(w, content, *rel_y)
            }
            SharedPrimitive::Shared { north, south } => {
                north.queue_print_shared_content(w, south, content)
            }
        }
    }

    pub fn with<'ret, 'cont: 'ret>(
        self,
        content: EitherContentRef<'cont>,
    ) -> PrimitiveWithContent<'ret>
    where
        'a: 'ret,
    {
        PrimitiveWithContent {
            primitive: self,
            content,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EitherContent {
    Styled(Styled<String>),
    Raw(String),
    Width(usize),
}
impl EitherContent {
    pub fn as_ref(&self) -> EitherContentRef<'_> {
        match self {
            EitherContent::Styled(Styled { content, style }) => EitherContentRef::Styled(Styled {
                content: content.as_str(),
                style: style.clone(),
            }),
            EitherContent::Raw(content) => EitherContentRef::Raw(content.as_str()),
            EitherContent::Width(w) => EitherContentRef::Width(*w),
        }
    }

    pub fn as_width(&'_ self) -> EitherContentRef<'_> {
        match self {
            EitherContent::Styled(content) => EitherContentRef::Width(content.length()),
            EitherContent::Raw(content) => EitherContentRef::Width(content.length()),
            EitherContent::Width(width) => EitherContentRef::Width(*width),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub enum EitherContentRef<'a> {
    Styled(Styled<&'a str>),
    Raw(&'a str),
    Width(usize),
}
impl<'a> EitherContentRef<'a> {
    pub fn as_width(&self) -> EitherContentRef<'a> {
        match self {
            EitherContentRef::Styled(content) => EitherContentRef::Width(content.length()),
            EitherContentRef::Raw(content) => EitherContentRef::Width(content.length()),
            EitherContentRef::Width(width) => EitherContentRef::Width(*width),
        }
    }
}
impl From<Styled<String>> for EitherContent {
    fn from(value: Styled<String>) -> Self {
        Self::Styled(value)
    }
}
impl From<String> for EitherContent {
    fn from(value: String) -> Self {
        Self::Raw(value)
    }
}
impl From<usize> for EitherContent {
    fn from(value: usize) -> Self {
        Self::Width(value)
    }
}

pub trait Printable: Display {
    fn queue_print<W: Write>(&self, w: &mut W) -> io::Result<()>;
    fn length(&self) -> usize;
}
impl Printable for usize {
    fn queue_print<W: Write>(&self, w: &mut W) -> io::Result<()> {
        for _ in 0..*self {
            queue_print!(w; ' ')?;
        }
        Ok(())
    }

    fn length(&self) -> usize {
        *self
    }
}
impl<'a> Printable for &'a str {
    fn queue_print<W: Write>(&self, w: &mut W) -> io::Result<()> {
        queue_print!(w; self)
    }

    fn length(&self) -> usize {
        self.chars().count()
    }
}
impl<'a> Printable for String {
    fn queue_print<W: Write>(&self, w: &mut W) -> io::Result<()> {
        queue_print!(w; self)
    }

    fn length(&self) -> usize {
        self.chars().count()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Styled<P: Printable> {
    content: P,
    style: ContentStyle,
}
impl<P: Printable> Display for Styled<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}
impl<P: Printable> Printable for Styled<P> {
    fn queue_print<W: Write>(&self, w: &mut W) -> io::Result<()> {
        queue_print!(w; self.style => &self.content)
    }

    fn length(&self) -> usize {
        self.content.length()
    }
}

mod connects {
    pub trait ConnectNorth {
        fn connect_north(&mut self);
        fn clear_north(&mut self);
        fn connected_north(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_north();
            s
        }
        fn cleared_north(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.clear_north();
            s
        }
    }
    pub trait ConnectEast {
        fn connect_east(&mut self);
        fn clear_east(&mut self);
        fn connected_east(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_east();
            s
        }
        fn cleared_east(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.clear_east();
            s
        }
    }
    pub trait ConnectSouth {
        fn connect_south(&mut self);
        fn clear_south(&mut self);
        fn connected_south(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_south();
            s
        }
        fn cleared_south(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.clear_south();
            s
        }
    }
    pub trait ConnectWest {
        fn connect_west(&mut self);
        fn clear_west(&mut self);
        fn connected_west(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_west();
            s
        }
        fn cleared_west(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.clear_west();
            s
        }
    }
    pub trait Passes {
        fn pass_horizontal(&mut self);
        fn passed_horizontal(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.pass_horizontal();
            s
        }
        fn clear_horizontal(&mut self);
        fn cleared_horizontal(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.clear_horizontal();
            s
        }

        fn pass_vertical(&mut self);
        fn passed_vertical(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.pass_vertical();
            s
        }
        fn clear_vertical(&mut self);
        fn cleared_vertical(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.clear_vertical();
            s
        }
    }
    impl<T: ConnectNorth + ConnectEast + ConnectSouth + ConnectWest> Passes for T {
        fn pass_horizontal(&mut self) {
            self.connect_east();
            self.connect_west();
        }
        fn pass_vertical(&mut self) {
            self.connect_north();
            self.connect_south();
        }
        fn clear_horizontal(&mut self) {
            self.clear_east();
            self.clear_west();
        }
        fn clear_vertical(&mut self) {
            self.clear_north();
            self.clear_south();
        }
    }
    pub trait Connects: ConnectNorth + ConnectEast + ConnectSouth + ConnectWest + Passes {}
    impl<T: ConnectNorth + ConnectEast + ConnectSouth + ConnectWest + Passes> Connects for T {}
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IsDirection {
        pub north: bool,
        pub east: bool,
        pub south: bool,
        pub west: bool,
    }
    #[rustfmt::skip]
    macro_rules! expand_direction {
        (North; N) => {true}; (North; E) => {false}; (North; S) => {false}; (North; W) => {false};
        (East; N) => {false}; (East; E) => {true}; (East; S) => {false}; (East; W) => {false};
        (South; N) => {false}; (South; E) => {false}; (South; S) => {true}; (South; W) => {false};
        (West; N) => {false}; (West; E) => {false}; (West; S) => {false}; (West; W) => {true};
        ($cc:ident; $($dir:ident)*) => {
            false $(|| expand_direction!($cc; $dir))*
        }
    }
    macro_rules! is_direction {
        ($($dir:ident)*) => {
            IsDirection {
                north: const {expand_direction!(North; $($dir)* )},
                east: const {expand_direction!(East; $($dir)* )},
                south: const {expand_direction!(South; $($dir)* )},
                west: const {expand_direction!(West; $($dir)* )},
            }
        };
    }
    pub trait HasDirection {
        fn is_direction(&self) -> IsDirection;
    }
    pub trait Combines<Rhs: HasDirection = Self>: Connects {
        fn combine(&mut self, rhs: &Rhs);
        fn combined(self, rhs: &Rhs) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.combine(rhs);
            s
        }
    }

    pub trait ExtendEast<Est = Self> {
        fn extend_east(&mut self, est: &mut Est);
    }
    pub trait ExtendSouth<Sth = Self> {
        fn extend_south(&mut self, sth: &mut Sth);
    }
    pub trait Extends<Nth: ExtendSouth<Self>, Wst: ExtendEast<Self>>: Sized {
        fn extend_from(&mut self, nth: &mut Nth, wst: &mut Wst) {
            nth.extend_south(self);
            wst.extend_east(self);
        }
    }
    impl<T, N: ExtendSouth<T>, W: ExtendEast<T>> Extends<N, W> for T {}

    impl ConnectNorth for char {
        #[inline(always)]
        fn connect_north(&mut self) {
            *self = match *self {
                ' ' => '╵',
                '╵' => '╵',
                '╶' => '└',
                '╷' => '│',
                '╴' => '┘',
                '└' => '└',
                '│' => '│',
                '┘' => '┘',
                '┌' => '├',
                '─' => '┴',
                '┐' => '┤',
                '├' => '├',
                '┴' => '┴',
                '┤' => '┤',
                '┬' => '┼',
                '┼' => '┼',
                c => c,
            }
        }
        #[inline(always)]
        fn clear_north(&mut self) {
            *self = match *self {
                '╵' => ' ',
                '└' => '╶',
                '│' => '╷',
                '┘' => '╴',
                '├' => '┌',
                '┴' => '─',
                '┤' => '┐',
                '┼' => '┬',
                c => c,
            }
        }
    }
    impl ConnectEast for char {
        #[inline(always)]
        fn connect_east(&mut self) {
            *self = match *self {
                ' ' => '╶',
                '╵' => '└',
                '╶' => '╶',
                '╷' => '┌',
                '╴' => '─',
                '└' => '└',
                '│' => '├',
                '┘' => '┴',
                '┌' => '┌',
                '─' => '─',
                '┐' => '┬',
                '├' => '├',
                '┴' => '┴',
                '┤' => '┼',
                '┬' => '┬',
                '┼' => '┼',
                c => c,
            }
        }
        #[inline(always)]
        fn clear_east(&mut self) {
            *self = match *self {
                '╶' => ' ',
                '└' => '╵',
                '┌' => '╷',
                '─' => '╴',
                '├' => '│',
                '┴' => '┘',
                '┬' => '┐',
                '┼' => '┤',
                c => c,
            }
        }
    }
    impl ConnectSouth for char {
        #[inline(always)]
        fn connect_south(&mut self) {
            *self = match *self {
                ' ' => '╷',
                '╵' => '│',
                '╶' => '┌',
                '╷' => '╷',
                '╴' => '┐',
                '└' => '├',
                '│' => '│',
                '┘' => '┤',
                '┌' => '┌',
                '─' => '┬',
                '┐' => '┐',
                '├' => '├',
                '┴' => '┼',
                '┤' => '┤',
                '┬' => '┬',
                '┼' => '┼',
                c => c,
            }
        }
        #[inline(always)]
        fn clear_south(&mut self) {
            *self = match *self {
                '╷' => ' ',
                '│' => '╵',
                '┌' => '╶',
                '┐' => '╴',
                '├' => '└',
                '┤' => '┘',
                '┬' => '─',
                '┼' => '┴',
                c => c,
            }
        }
    }
    impl ConnectWest for char {
        #[inline(always)]
        fn connect_west(&mut self) {
            *self = match *self {
                ' ' => '╴',
                '╵' => '┘',
                '╶' => '─',
                '╷' => '┐',
                '╴' => '╴',
                '└' => '┴',
                '│' => '┤',
                '┘' => '┘',
                '┌' => '┬',
                '─' => '─',
                '┐' => '┐',
                '├' => '┼',
                '┴' => '┴',
                '┤' => '┤',
                '┬' => '┬',
                '┼' => '┼',
                c => c,
            }
        }
        #[inline(always)]
        fn clear_west(&mut self) {
            *self = match *self {
                '╴' => ' ',
                '┘' => '╵',
                '─' => '╶',
                '┐' => '╷',
                '┴' => '└',
                '┤' => '│',
                '┬' => '┌',
                '┼' => '├',
                c => c,
            }
        }
    }
    impl HasDirection for char {
        #[inline(always)]
        fn is_direction(&self) -> IsDirection {
            match self {
                ' ' => is_direction!(),
                '╵' => is_direction!(N),
                '╶' => is_direction!(E),
                '╷' => is_direction!(S),
                '╴' => is_direction!(W),
                '└' => is_direction!(N E),
                '│' => is_direction!(N S),
                '┘' => is_direction!(N W),
                '┌' => is_direction!(E S),
                '─' => is_direction!(E W),
                '┐' => is_direction!(S W),
                '├' => is_direction!(N E S),
                '┴' => is_direction!(N E W),
                '┤' => is_direction!(N S W),
                '┬' => is_direction!(E S W),
                '┼' => is_direction!(N E S W),
                _ => panic!("{} does not have a direction", self),
            }
        }
    }
    impl Combines for char {
        #[inline(always)]
        fn combine(&mut self, rhs: &Self) {
            let is_dir = rhs.is_direction();
            if is_dir.north {
                self.connect_north();
            }
            if is_dir.east {
                self.connect_east();
            }
            if is_dir.south {
                self.connect_south();
            }
            if is_dir.west {
                self.connect_west();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Write, stdout};

    use crate::{
        circuit::Circuit,
        debug_simulator::DebugSimulator,
        debug_terminal::show_circuit::{Column, Primitive, connects::ExtendEast, show_circuit},
        gate::{Gate, GateType, QBits},
        instruction::Instruction,
        simulator::BuildSimulator,
    };

    #[test]
    fn rep_i() {
        for bw in (5..16).step_by(2) {
            assert_eq!(Primitive::rep_mid_i(bw, 0), 0);
            for i in 1..(bw / 2) {
                assert_eq!(Primitive::rep_mid_i(bw, i), 1);
            }
            assert_eq!(Primitive::rep_mid_i(bw, bw / 2), 2);
            for i in (bw / 2 + 1)..(bw - 1) {
                assert_eq!(Primitive::rep_mid_i(bw, i), 3);
            }
            assert_eq!(Primitive::rep_mid_i(bw, bw - 1), 4);

            for i in 0..(bw / 2) {
                assert_eq!(Primitive::rep_sides_i(bw, i), 0);
            }
            assert_eq!(Primitive::rep_sides_i(bw, bw / 2), 1);
            for i in (bw / 2 + 1)..(bw) {
                assert_eq!(Primitive::rep_sides_i(bw, i), 2);
            }

            for i in 0..(bw / 2 - 1) {
                assert_eq!(Primitive::rep_sides_3_mid_i(bw, i), 0);
            }
            assert_eq!(Primitive::rep_sides_3_mid_i(bw, bw / 2 - 1), 1);
            assert_eq!(Primitive::rep_sides_3_mid_i(bw, bw / 2), 2);
            assert_eq!(Primitive::rep_sides_3_mid_i(bw, bw / 2 + 1), 3);
            for i in (bw / 2 + 2)..(bw) {
                assert_eq!(Primitive::rep_sides_3_mid_i(bw, i), 4);
            }
        }
    }

    #[test]
    #[allow(unreachable_code)]
    fn column() {
        return;
        let w = &mut stdout();
        let mut col = Column::init_with_gate(String::from("U"));
        // let mut col = Column::init_with_track();
        // col.extend_with_gate();
        // col.extend_with_gate();
        col.extend_with_track();
        col.extend_with_track();
        col.extend_with_gate();
        col.extend_with_gate();
        col.extend_with_gate();
        col.extend_with_gate();
        col.extend_with_track();
        for printme in col.into_iter() {
            printme.queue_print(w).unwrap();
            queue_print!(w; "\r\n").unwrap();
        }
        w.flush().unwrap();
    }

    #[test]
    #[allow(unreachable_code)]
    fn measurement() {
        return;
        let w = &mut stdout();
        let instruction = Instruction::Measurement(QBits::from_bitstring(0b101011010), "".into());
        let mut track_col = Column::only_tracks(10);
        let mut measure_col = Column::from_instruction(10, &instruction);
        track_col.extend_east(&mut measure_col);
        for (tp, mp) in track_col.into_iter().zip(measure_col.into_iter()) {
            tp.queue_print(w).unwrap();
            mp.queue_print(w).unwrap();
            queue_print!(w; "\r\n").unwrap();
        }
        w.flush().unwrap();
    }

    #[test]
    #[allow(unreachable_code)]
    fn with_gate() {
        return;
        let w = &mut stdout();

        let instruction1 = Instruction::Gate(Gate::new(GateType::H, &[1, 5, 6], &[3]).unwrap());
        let instruction2 = Instruction::Gate(Gate::new(GateType::Y, &[], &[1]).unwrap());
        let instruction3 = Instruction::Gate(Gate::new(GateType::X, &[1, 5], &[2]).unwrap());
        let instruction4 = Instruction::Gate(Gate::new(GateType::Z, &[1], &[4]).unwrap());

        let mut col0 = Column::only_tracks(7);
        let mut col1 = Column::from_instruction(7, &instruction1);
        let mut col2 = Column::from_instruction(7, &instruction2);
        let mut col3 = Column::from_instruction(7, &instruction3);
        let mut col4 = Column::from_instruction(7, &instruction4);
        let mut col5 = Column::only_tracks(7);
        let mut col6 = Column::from_instruction(
            7,
            &Instruction::Measurement(QBits::from_bitstring(0xFFFF), "".into()),
        );
        let mut col7 = Column::only_tracks(7);

        col0.extend_east(&mut col1);
        col1.extend_east(&mut col2);
        col2.extend_east(&mut col3);
        col3.extend_east(&mut col4);
        col4.extend_east(&mut col5);
        col5.extend_east(&mut col6);
        col6.extend_east(&mut col7);

        for (((((((c0, c1), c2), c3), c4), c5), c6), c7) in col0
            .into_iter()
            .zip(col1.into_iter())
            .zip(col2.into_iter())
            .zip(col3.into_iter())
            .zip(col4.into_iter())
            .zip(col5.into_iter())
            .zip(col6.into_iter())
            .zip(col7.into_iter())
        {
            c0.queue_print(w).unwrap();
            c1.queue_print(w).unwrap();
            c2.queue_print(w).unwrap();
            c3.queue_print(w).unwrap();
            c4.queue_print(w).unwrap();
            c5.queue_print(w).unwrap();
            c6.queue_print(w).unwrap();
            c7.queue_print(w).unwrap();
            queue_print!(w; "\r\n").unwrap();
        }
        w.flush().unwrap();
    }

    #[test]
    #[allow(unreachable_code)]
    fn circuit() {
        return;
        let w = &mut stdout();

        let circuit = Circuit::new(7)
            .h(2)
            .z(5)
            .cx(&[1], 3)
            .x(6)
            .y(2)
            .swap(3, 5)
            .cswap(&[2], 3, 4);
        let sim = DebugSimulator::build(circuit).unwrap();
        show_circuit(w, &sim).unwrap();
    }
}
