use std::{
    fmt::Display,
    io::{self, Write},
    marker::PhantomData,
};

use crate::simulator::{DebuggableSimulator, StoredCircuitSimulator};

// type Block = [[char; BLOCK_SIZE.width]; BLOCK_SIZE.height];
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Char<const C: char>;
pub trait BlockTrait {
    fn char<const X: usize, const Y: usize>(&self) -> char;
}
pub trait ColumnTrait {
    const HEIGHT: usize;
    fn block(&self, rel_lane: usize) -> Option<&impl BlockTrait>;
}
#[rustfmt::skip]
#[derive(Debug, Clone, Copy)]
pub struct Block([[char; 3]; 3]);
const fn sblock() -> Block {
    Block([[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']])
}
#[rustfmt::skip]
impl BlockTrait for Block {
    #[rustfmt::skip]
    #[inline(always)]
    fn char<const X: usize, const Y: usize>(&self) -> char {
        debug_assert!(X < 3 && Y < 3);
        self.0[Y][X]
    }
}
// #[rustfmt::skip]
// #[derive(Debug, Default, Clone, Copy)]
// pub struct Column<T>(T);
// impl<B: BlockTrait> ColumnTrait for Column<B> {
//     const HEIGHT: usize = 1;
//     fn block(&self, rel_lane: usize) -> Option<&impl BlockTrait> {
//         if rel_lane == 0 {
//             Some(&self.0)
//         } else {
//             None
//         }
//     }
// }
// impl<C: ColumnTrait, B: BlockTrait> ColumnTrait for Column<(C, B)> {
//     const HEIGHT: usize = C::HEIGHT + 1;
//     fn block(&self, rel_lane: usize) -> Option<&impl BlockTrait> {
//         if rel_lane == Self::HEIGHT - 1 {
//             Some(&self.0.1)
//         } else if rel_lane < Self::HEIGHT - 1{
//             self.0.0.block(rel_lane)
//         } else {
//             None
//         }
//     }
// }

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
    use crate::debug_terminal::show_circuit::sblock;

    use super::{Block, BlockTrait, Char, Pos, Size};
    use std::{fmt::Display, io::stdout, marker::PhantomData, ops::Deref};

    // https://en.wikipedia.org/wiki/Box-drawing_characters
    // https://www.alt-codes.net/bullet_alt_codes.php
    // https://www.compart.com/en/unicode/block/U+2B00

    #[rustfmt::skip]
    #[derive(Debug, Clone, Copy)]
    pub struct Track(Block);
    #[rustfmt::skip]
    const fn strack() -> Track {
        Track(Block([
            [' ', '│', ' '],
            ['─', '┼', '─'],
            [' ', '│', ' ']
        ]))
    }
    #[rustfmt::skip]
    impl BlockTrait for Track {
        #[rustfmt::skip]
        #[inline(always)]
        fn char<const X: usize, const Y: usize>(&self) -> char {
            self.0.char::<X, Y>()
        }
    }
    #[rustfmt::skip]
    #[derive(Debug, Clone, Copy)]
    pub struct Gate(Block);
    #[rustfmt::skip]
    const fn sgate() -> Gate {
        Gate(Block([
            ['┌', '─', '┐'],
            ['│', ' ', '│'],
            ['└', '─', '┘']
        ]))
    }
    #[rustfmt::skip]
    impl BlockTrait for Gate {
        #[rustfmt::skip]
        #[inline(always)]
        fn char<const X: usize, const Y: usize>(&self) -> char {
            self.0.char::<X, Y>()
        }
    }

    pub trait ConnectNorth {
        fn connect_north(&mut self);
        fn connected_north(self) -> Self where Self: Sized {
            let mut s = self;
            s.connect_north();
            s
        }
    }
    pub trait ConnectEast {
        fn connect_east(&mut self);
        fn connected_east(self) -> Self where Self: Sized {
            let mut s = self;
            s.connect_east();
            s
        }
    }
    pub trait ConnectSouth {
        fn connect_south(&mut self);
        fn connected_south(self) -> Self where Self: Sized {
            let mut s = self;
            s.connect_south();
            s
        }
    }
    pub trait ConnectWest {
        fn connect_west(&mut self);
        fn connected_west(self) -> Self where Self: Sized {
            let mut s = self;
            s.connect_west();
            s
        }
    }
    pub trait Passes {
        fn pass_horizontal(&mut self);
        fn passed_horizontal(self) -> Self where Self: Sized {
            let mut s = self;
            s.pass_horizontal();
            s
        }
        fn pass_vertical(&mut self);
        fn passed_vertical(self) -> Self where Self: Sized {
            let mut s = self;
            s.pass_vertical();
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
    }
    pub trait Connects: ConnectNorth + ConnectEast + ConnectSouth + ConnectWest + Passes {}
    impl<T: ConnectNorth + ConnectEast + ConnectSouth + ConnectWest + Passes> Connects for T {}
    pub struct IsDirection {
        north: bool,
        east: bool,
        south: bool,
        west: bool,
    }
    pub trait HasDirection {
        fn is_direction(&self) -> IsDirection;
    }
    pub trait Combines<Rhs: HasDirection = Self>: Connects {
        type Output: Connects;
        fn combine(self, rhs: Rhs) -> Self::Output;
    }
    impl ConnectNorth for char {
        fn connect_north(&mut self) {
            *self = match self {
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
                _ => panic!("Cannot connect {} north", self),
            }
        }
    }
    impl ConnectEast for char {
        fn connect_east(&mut self) {
            *self = match self {
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
                _ => panic!("Cannot connect {} north", self),
            }
        }
    }
    impl ConnectSouth for char {
        fn connect_south(&mut self) {
            *self = match self {
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
                _ => panic!("Cannot connect {} north", self),
            }
        }
    }
    impl ConnectWest for char {
        fn connect_west(&mut self) {
            *self = match self {
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
                _ => panic!("Cannot connect {} north", self),
            };
        }
    }
    #[rustfmt::skip]
    impl ConnectNorth for Track {
        fn connect_north(&mut self){
            let [
                [nw, nn, ne],
                [ww, cc, ee],
                [sw, ss, se]
            ] = &mut self.0.0;
            nn.pass_vertical();
            cc.connect_north();
        }
    }
    #[rustfmt::skip]
    impl ConnectEast for  Track {
        fn connect_east(&mut self){
            let [
                [nw, nn, ne],
                [ww, cc, ee],
                [sw, ss, se]
            ] = &mut self.0.0;
            ee.pass_horizontal();
            cc.connect_east();
        }
    }
    #[rustfmt::skip]
    impl ConnectSouth for Track {
        fn connect_south(&mut self)  {
            let [
                [nw, nn, ne],
                [ww, cc, ee],
                [sw, ss, se]
            ] = &mut self.0.0;
            ss.pass_vertical();
            cc.connect_north();
        }
    }
    #[rustfmt::skip]
    impl ConnectWest for  Track {
        fn connect_west(&mut self) {
            let [
                [nw, nn, ne],
                [ww, cc, ee],
                [sw, ss, se]
            ] = &mut self.0.0;
            let ww = ww.pass_horizontal();
            let cc = cc.connect_east();
        }
    }

    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectNorth for Track<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type NOutput = Track<
    //         NW, NN::VOutput, NE,
    //         WW, CC::NOutput, EE,
    //         SW, SS,          SE,
    //     >;
    //     const NRESULT: Self::NOutput = strack();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectEast for Track<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type EOutput = Track<
    //         NW, NN,          NE,
    //         WW, CC::EOutput, EE::HOutput,
    //         SW, SS,          SE,
    //     >;
    //     const ERESULT: Self::EOutput = strack();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectSouth for Track<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type SOutput = Track<
    //         NW, NN,          NE,
    //         WW, CC::SOutput, EE,
    //         SW, SS::VOutput, SE,
    //     >;
    //     const SRESULT: Self::SOutput = strack();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectWest for Track<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type WOutput = Track<
    //         NW,          NN,          NE,
    //         WW::HOutput, CC::WOutput, EE,
    //         SW,          SS,          SE,
    //     >;
    //     const WRESULT: Self::WOutput = strack();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > Connects for Track<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     const SELF: Self = strack();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectNorth for Gate<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type NOutput = Gate<
    //         NW, NN::VOutput, NE,
    //         WW, CC,          EE,
    //         SW, SS,          SE,
    //     >;
    //     const NRESULT: Self::NOutput = sgate();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectEast for Gate<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type EOutput = Gate<
    //         NW, NN, NE,
    //         WW, CC, EE::HOutput,
    //         SW, SS, SE,
    //     >;
    //     const ERESULT: Self::EOutput = sgate();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectSouth for Gate<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type SOutput = Gate<
    //         NW, NN,          NE,
    //         WW, CC,          EE,
    //         SW, SS::VOutput, SE,
    //     >;
    //     const SRESULT: Self::SOutput = sgate();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > ConnectWest for Gate<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     type WOutput = Gate<
    //         NW,          NN, NE,
    //         WW::HOutput, CC, EE,
    //         SW,          SS, SE,
    //     >;
    //     const WRESULT: Self::WOutput = sgate();
    // }
    // #[rustfmt::skip]
    // impl<
    //     NW: Connects, NN: Connects, NE: Connects,
    //     WW: Connects, CC: Connects, EE: Connects,
    //     SW: Connects, SS: Connects, SE: Connects,
    // > Connects for Gate<
    //     NW, NN, NE,
    //     WW, CC, EE,
    //     SW, SS, SE,
    // > {
    //     const SELF: Self = sgate();
    // }

    // struct ConnectNorth<const C: char>;
    // #[rustfmt::skip]
    // impl ConnectNorth<' '> {const RESULT: char = ' '}
    // #[inline(always)]
    // const fn ctrl(block: Block) -> Block {
    //     let mut block = block;
    //     block[1][1] = '⬤';
    //     block
    // }

    // #[inline(always)]
    // const fn cinv(block: Block) -> Block {
    //     let mut block = block;
    //     block[1][1] = '⭘';
    //     block
    // }

    // #[inline(always)]
    // const fn cnot(block: Block) -> Block {
    //     let mut block = block;
    //     block[1][1] = '⨁';
    //     block
    // }

    // #[inline(always)]
    // const fn connect_north(block: Block) -> Block {
    //     let mut block = block;
    //     block[0][1] = '┴';
    //     block
    // }
    // #[inline(always)]
    // const fn connect_east(block: Block) -> Block {
    //     let mut block = block;
    //     block[1][2] = '├';
    //     block
    // }
    // #[inline(always)]
    // const fn connect_south(block: Block) -> Block {
    //     let mut block = block;
    //     block[2][1] = '┬';
    //     block
    // }
    // #[inline(always)]
    // const fn connect_west(block: Block) -> Block {
    //     let mut block = block;
    //     block[1][0] = '┤';
    //     block
    // }

    pub mod track {
        // use super::{Block, Char, NULL};

        // type OOOO = NULL;

        // def_block!(OOOW,
        //     ' ' ' ' ' ';
        //     '─' '─' ' ';
        //     ' ' ' ' ' '
        // );
        // def_block!(OOSO,
        //     ' ' ' ' ' ';
        //     ' ' '│' ' ';
        //     ' ' '│' ' '
        // );
        // def_block!(OEOO,
        //     ' ' ' ' ' ';
        //     ' ' '─' '─';
        //     ' ' ' ' ' '
        // );
        // def_block!(NOOO,
        //     ' ' '│' ' ';
        //     ' ' '│' ' ';
        //     ' ' ' ' ' '
        // );
        // const_block!(NOSO,
        //     ' ' '│' ' ';
        //     ' ' '│' ' ';
        //     ' ' '│' ' '
        // );
        // const_block!(OEOW,
        //     ' ' ' ' ' ';
        //     '─' '─' '─';
        //     ' ' ' ' ' '
        // );
        // const_block!(OOSW,
        //     ' ' ' ' ' ';
        //     '─' '┐' ' ';
        //     ' ' '│' ' '
        // );
        // const_block!(OESO,
        //     ' ' ' ' ' ';
        //     ' ' '┌' '─';
        //     ' ' '│' ' '
        // );
        // const_block!(NOOW,
        //     ' ' '│' ' ';
        //     '─' '┘' ' ';
        //     ' ' ' ' ' '
        // );
        // const_block!(NEOO,
        //     ' ' '│' ' ';
        //     ' ' '└' '─';
        //     ' ' ' ' ' '
        // );
        // const_block!(OESW,
        //     ' ' ' ' ' ';
        //     '─' '┬' '─';
        //     ' ' '│' ' '
        // );
        // const_block!(NOSW,
        //     ' ' '│' ' ';
        //     '─' '┤' ' ';
        //     ' ' '│' ' '
        // );
        // const_block!(NEOW,
        //     ' ' '│' ' ';
        //     '─' '┴' '─';
        //     ' ' ' ' ' '
        // );
        // const_block!(NESO,
        //     ' ' '│' ' ';
        //     ' ' '├' '─';
        //     ' ' '│' ' '
        // );
        // const_block!(NESW,
        //     ' ' '│' ' ';
        //     '─' '┼' '─';
        //     ' ' '│' ' '
        // );

        // const_block!(NESW_PASS,
        //     ' ' '│' ' ';
        //     '⬤' '│' '⬤';
        //     ' ' '│' ' '
        // );
    }

    pub mod gate {
        use super::Block;

        // const_block!(NL,
        //     ' ' ' ' ' ';
        //     ' ' ' ' ' ';
        //     ' ' ' ' ' '
        // );
        // const_block!(FL,
        //     '┌' '─' '┐';
        //     '│' ' ' '│';
        //     '└' '─' '┘'
        // );
        // const_block!(NW,
        //     '┌' '─' '─';
        //     '│' ' ' ' ';
        //     '│' ' ' ' '
        // );
        // const_block!(N,
        //     '─' '─' '─';
        //     ' ' ' ' ' ';
        //     ' ' ' ' ' '
        // );
        // const_block!(NN,
        //     '┌' '─' '┐';
        //     '│' ' ' '│';
        //     '│' ' ' '│'
        // );
        // const_block!(NE,
        //     '─' '─' '┐';
        //     ' ' ' ' '│';
        //     ' ' ' ' '│'
        // );
        // const_block!(E,
        //     ' ' ' ' '│';
        //     ' ' ' ' '│';
        //     ' ' ' ' '│'
        // );
        // const_block!(EE,
        //     '─' '─' '┐';
        //     ' ' ' ' '│';
        //     '─' '─' '┘'
        // );
        // const_block!(SE,
        //     ' ' ' ' '│';
        //     ' ' ' ' '│';
        //     '─' '─' '┘'
        // );
        // const_block!(S,
        //     ' ' ' ' ' ';
        //     ' ' ' ' ' ';
        //     '─' '─' '─'
        // );
        // const_block!(SS,
        //     '│' ' ' '│';
        //     '│' ' ' '│';
        //     '└' '─' '┘'
        // );
        // const_block!(SW,
        //     '│' ' ' ' ';
        //     '│' ' ' ' ';
        //     '└' '─' '─'
        // );
        // const_block!(W,
        //     '│' ' ' ' ';
        //     '│' ' ' ' ';
        //     '│' ' ' ' '
        // );
        // const_block!(WW,
        //     '┌' '─' '─';
        //     '│' ' ' ' ';
        //     '└' '─' '─'
        // );
        // const_block!(WE,
        //     '│' ' ' '│';
        //     '│' ' ' '│';
        //     '│' ' ' '│'
        // );
        // const_block!(SK,
        //     ' ' ' ' ' ';
        //     '⋮' ' ' '⋮';
        //     ' ' ' ' ' '
        // );
    }
}
#[cfg(test)]
mod tests {
    use crate::debug_terminal::show_circuit::{Char, block::Connects};

    #[test]
    fn combine() {
        // assert_eq!(
        // <Char<'┌'> as Combine<Char<'│'>>>::CRESULT,
        // <Char::<'├'> as Connects>::SELF
        // );
    }
}
