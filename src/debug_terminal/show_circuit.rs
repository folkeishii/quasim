use std::{
    fmt::Display,
    io::{self, Write},
    marker::PhantomData,
};

use crate::simulator::{DebuggableSimulator, StoredCircuitSimulator};

// type Block = [[char; BLOCK_SIZE.width]; BLOCK_SIZE.height];

const BLOCK_SIZE: usize = 5;


#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Char<const C: char>;
pub trait BlockTrait {
    fn char<const X: usize, const Y: usize>(&self) -> char;
}
pub trait ColumnTrait {
    fn block(&self, rel_lane: usize) -> Option<&impl BlockTrait>;
}
#[rustfmt::skip]
#[derive(Debug, Clone, Copy)]
pub struct Block([[char; 3]; 3]);
#[rustfmt::skip]
impl Block {
    pub const fn mid_i<const I: usize, const SIZE: usize>() -> usize {
        //  0  1  2  3  4 // (-SIZE as isize / 2)
        // -2 -1  0  1  2 // (/2)
        // -1  0  0  0  1 // (+1)
        //  0  1  1  1  2
        const {((I as isize - SIZE as isize / 2) / 2 + 1) as usize}
    }
    pub const fn sides_i<const I: usize, const SIZE: usize>() -> usize {
        //  0  1  2  3  4 // (-SIZE as isize / 2)
        // -2 -1  0  1  2 // (signum)
        // -1 -1  0  1  1 // (+1)
        //  0  0  1  2  2
        const {((I as isize - SIZE as isize / 2).signum() + 1) as usize}
    }
    #[inline(always)]
    pub fn char_repeat_mid<const X: usize, const Y: usize>(&self) -> char {
        assert!(X < BLOCK_SIZE && Y < BLOCK_SIZE);
        self.0[Self::mid_i::<Y, BLOCK_SIZE>()][Self::mid_i::<X, BLOCK_SIZE>()]
    }
    #[inline(always)]
    pub fn char_repeat_sides<const X: usize, const Y: usize>(&self) -> char {
        assert!(X < BLOCK_SIZE && Y < BLOCK_SIZE);
        self.0[Self::sides_i::<Y, BLOCK_SIZE>()][Self::sides_i::<X, BLOCK_SIZE>()]
    }
}
const fn sblock() -> Block {
    Block([[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']])
}
// #[rustfmt::skip]
// impl BlockTrait for Block {
//     #[rustfmt::skip]
//     #[inline(always)]
//     fn char<const X: usize, const Y: usize>(&self) -> char {
//         debug_assert!(X < 3 && Y < 3);
//         self.0[Y][X]
//     }
// }
// #[rustfmt::skip]
// #[derive(Debug, Default, Clone, Copy)]
// pub struct Column<B, C>(B, C);
// impl<B: BlockTrait> ColumnTrait for B {
//     fn block(&self, rel_lane: usize) -> Option<&impl BlockTrait> {
//         if rel_lane == 0 {
//             Some(self)
//         } else {
//             None
//         }
//     }
// }
// impl<B: BlockTrait, C: ColumnTrait> ColumnTrait for Column<(B, C)> {
//     fn block(&self, rel_lane: usize) -> Option<&impl BlockTrait> {
//         if rel_lane == 0 {
//             Some(&self.0.0)
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
mod block {
    use crate::debug_terminal::show_circuit::sblock;

    use super::{Block, BlockTrait};
    use std::{fmt::Display, io::stdout, marker::PhantomData, ops::Deref};

    // https://en.wikipedia.org/wiki/Box-drawing_characters
    // https://www.alt-codes.net/bullet_alt_codes.php
    // https://www.compart.com/en/unicode/block/U+2B00

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
    impl BlockTrait for Track {
        #[inline(always)]
        fn char<const X: usize, const Y: usize>(&self) -> char {
            self.0.char_repeat_sides::<X, Y>()
        }
    }
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
    impl BlockTrait for Gate {
        #[inline(always)]
        fn char<const X: usize, const Y: usize>(&self) -> char {
            self.0.char_repeat_mid::<X, Y>()
        }
    }
    pub struct Widened<const W: usize, B: BlockTrait>(B);
    impl<const W: usize, B: BlockTrait> BlockTrait for Widened<W, B> {
        #[inline(always)]
        fn char<const X: usize, const Y: usize>(&self) -> char {
            self.0.char_repeat_mid::<X, Y>()
        }
    }
    pub trait ConnectNorth {
        fn connect_north(&mut self);
        fn connected_north(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_north();
            s
        }
    }
    pub trait ConnectEast {
        fn connect_east(&mut self);
        fn connected_east(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_east();
            s
        }
    }
    pub trait ConnectSouth {
        fn connect_south(&mut self);
        fn connected_south(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_south();
            s
        }
    }
    pub trait ConnectWest {
        fn connect_west(&mut self);
        fn connected_west(self) -> Self
        where
            Self: Sized,
        {
            let mut s = self;
            s.connect_west();
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
        fn pass_vertical(&mut self);
        fn passed_vertical(self) -> Self
        where
            Self: Sized,
        {
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
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IsDirection {
        north: bool,
        east: bool,
        south: bool,
        west: bool,
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
    impl ConnectNorth for char {
        #[inline(always)]
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
        #[inline(always)]
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
                _ => panic!("Cannot connect {} east", self),
            }
        }
    }
    impl ConnectSouth for char {
        #[inline(always)]
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
                _ => panic!("Cannot connect {} south", self),
            }
        }
    }
    impl ConnectWest for char {
        #[inline(always)]
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
                _ => panic!("Cannot connect {} west", self),
            };
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
    #[rustfmt::skip]
    impl ConnectNorth for Track {
        #[inline(always)]
        fn connect_north(&mut self){
            let [
                [_, nn, _],
                [_, cc, _],
                [_, _,  _]
            ] = &mut self.0.0;
            nn.pass_vertical();
            cc.connect_north();
        }
    }
    #[rustfmt::skip]
    impl ConnectEast for  Track {
        fn connect_east(&mut self){
            let [
                [_, _,  _],
                [_, cc, ee],
                [_, _,  _]
            ] = &mut self.0.0;
            ee.pass_horizontal();
            cc.connect_east();
        }
    }
    #[rustfmt::skip]
    impl ConnectSouth for Track {
        #[inline(always)]
        fn connect_south(&mut self)  {
            let [
                [_, _,  _],
                [_, cc, _],
                [_, ss, _]
            ] = &mut self.0.0;
            ss.pass_vertical();
            cc.connect_south();
        }
    }
    #[rustfmt::skip]
    impl ConnectWest for  Track {
        #[inline(always)]
        fn connect_west(&mut self) {
            let [
                [_,  _,  _],
                [ww, cc, _],
                [_,  _,  _]
            ] = &mut self.0.0;
            ww.pass_horizontal();
            cc.connect_west();
        }
    }
    #[rustfmt::skip]
    impl ConnectNorth for Gate {
        #[inline(always)]
        fn connect_north(&mut self){
            let [
                [_, nn, _],
                [_, _,  _],
                [_, _,  _]
            ] = &mut self.0.0;
            nn.connect_north();
        }
    }
    #[rustfmt::skip]
    impl ConnectEast for  Gate {
        fn connect_east(&mut self){
            let [
                [_, _,  _],
                [_, _,  ee],
                [_, _,  _]
            ] = &mut self.0.0;
            ee.connect_east();
        }
    }
    #[rustfmt::skip]
    impl ConnectSouth for Gate {
        #[inline(always)]
        fn connect_south(&mut self)  {
            let [
                [_, _,  _],
                [_, _,  _],
                [_, ss, _]
            ] = &mut self.0.0;
            ss.connect_north();
        }
    }
    #[rustfmt::skip]
    impl ConnectWest for  Gate {
        #[inline(always)]
        fn connect_west(&mut self) {
            let [
                [_,  _,  _],
                [ww, _,  _],
                [_,  _,  _]
            ] = &mut self.0.0;
            ww.connect_east();
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
    use std::io::stdout;

    use crate::debug_terminal::show_circuit::{Char, block::Connects};

    #[test]
    fn combine() {
        println!(stdout(); "┌───┐");
        println!(stdout(); "│ U │");
        println!(stdout(); "└───┘");
        // assert_eq!(
        // <Char<'┌'> as Combine<Char<'│'>>>::CRESULT,
        // <Char::<'├'> as Connects>::SELF
        // );
    }
}
