use std::{
    fmt::Display,
    io::{self, Write},
    marker::PhantomData,
};

use crate::simulator::{DebuggableSimulator, StoredCircuitSimulator};

// type Block = [[char; BLOCK_SIZE.width]; BLOCK_SIZE.height];
#[derive(Debug, Default, Clone, Copy)]
struct Char<const C: char>;
pub trait BlockTrait {
    fn char<const X: usize, const Y: usize>(&self) -> char;
}
#[rustfmt::skip]
#[derive(Debug, Default, Clone, Copy)]
pub struct Block<
    NW, NN, NE,
    WW, CC, EE,
    SW, SS, SE,
>(PhantomData<((NW, NN, NE), (WW, CC, EE), (SW, SS, SE))>);
#[rustfmt::skip]
impl <
    const NW: char, const NN: char, const NE: char,
    const WW: char, const CC: char, const EE: char,
    const SW: char, const SS: char, const SE: char,
> BlockTrait for Block<
    Char<NW>, Char<NN>, Char<NE>,
    Char<WW>, Char<CC>, Char<EE>,
    Char<SW>, Char<SS>, Char<SE>,
> {
    #[rustfmt::skip]
    #[inline(always)]
    fn char<const X: usize, const Y: usize>(&self) -> char {
        match (Y, X) {
            (0, 0) => NW, (0, 1) => NN, (0, 2) => NW,
            (1, 0) => WW, (1, 1) => CC, (1, 2) => EE,
            (2, 0) => SW, (2, 1) => SS, (2, 2) => SE,
            _ => panic!("({}, {}) is out of bounds", X, Y)
        }
    }
}
const fn sblock<NW, NN, NE, WW, CC, EE, SW, SS, SE>() -> Block<NW, NN, NE, WW, CC, EE, SW, SS, SE> {
    Block::<NW, NN, NE, WW, CC, EE, SW, SS, SE>(PhantomData)
}

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
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Track<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    >(Block::<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE
    >);
    #[rustfmt::skip]
    const fn strack<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    >() -> Track<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    > {
        Track(sblock())
    }
    #[rustfmt::skip]
    impl <
        const NW: char, const NN: char, const NE: char,
        const WW: char, const CC: char, const EE: char,
        const SW: char, const SS: char, const SE: char,
    > BlockTrait for Track<
        Char<NW>, Char<NN>, Char<NE>,
        Char<WW>, Char<CC>, Char<EE>,
        Char<SW>, Char<SS>, Char<SE>,
    > {
        #[rustfmt::skip]
        #[inline(always)]
        fn char<const X: usize, const Y: usize>(&self) -> char {
            match (Y, X) {
                (0, 0) => NW, (0, 1) => NN, (0, 2) => NW,
                (1, 0) => WW, (1, 1) => CC, (1, 2) => EE,
                (2, 0) => SW, (2, 1) => SS, (2, 2) => SE,
                _ => panic!("({}, {}) is out of bounds", X, Y)
            }
        }
    }
    #[rustfmt::skip]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Gate<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    >(Block::<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE
    >);
    #[rustfmt::skip]
    const fn sgate<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    >() -> Gate<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    > {
        Gate(sblock())
    }
    #[rustfmt::skip]
    impl <
        const NW: char, const NN: char, const NE: char,
        const WW: char, const CC: char, const EE: char,
        const SW: char, const SS: char, const SE: char,
    > BlockTrait for Gate<
        Char<NW>, Char<NN>, Char<NE>,
        Char<WW>, Char<CC>, Char<EE>,
        Char<SW>, Char<SS>, Char<SE>,
    > {
        #[rustfmt::skip]
        #[inline(always)]
        fn char<const X: usize, const Y: usize>(&self) -> char {
            match (Y, X) {
                (0, 0) => NW, (0, 1) => NN, (0, 2) => NW,
                (1, 0) => WW, (1, 1) => CC, (1, 2) => EE,
                (2, 0) => SW, (2, 1) => SS, (2, 2) => SE,
                _ => panic!("({}, {}) is out of bounds", X, Y)
            }
        }
    }

    pub const fn connect_north<C: ConnectNorth>() -> C::NOutput {
        C::NRESULT
    }
    pub const fn connect_east<C: ConnectEast>() -> C::EOutput {
        C::ERESULT
    }
    pub const fn connect_south<C: ConnectSouth>() -> C::SOutput {
        C::SRESULT
    }
    pub const fn connect_west<C: ConnectWest>() -> C::WOutput {
        C::WRESULT
    }
    pub const fn pass_horizontal<P: Passes>() -> P::HOutput {
        P::HRESULT
    }
    pub const fn pass_vertical<P: Passes>() -> P::VOutput {
        P::VRESULT
    }
    pub trait ConnectNorth {
        type NOutput: Connects;
        const NRESULT: Self::NOutput;
    }
    pub trait ConnectEast {
        type EOutput: Connects;
        const ERESULT: Self::EOutput;
    }
    pub trait ConnectSouth {
        type SOutput: Connects;
        const SRESULT: Self::SOutput;
    }
    pub trait ConnectWest {
        type WOutput: Connects;
        const WRESULT: Self::WOutput;
    }
    pub trait Passes {
        type HOutput: Connects;
        type VOutput: Connects;
        const HRESULT: Self::HOutput;
        const VRESULT: Self::VOutput;
    }
    impl<T: ConnectNorth + ConnectEast + ConnectSouth + ConnectWest> Passes for T {
        type HOutput = <T::EOutput as ConnectWest>::WOutput;
        type VOutput = <T::NOutput as ConnectSouth>::SOutput;
        const HRESULT: Self::HOutput = <T::EOutput as ConnectWest>::WRESULT;
        const VRESULT: Self::VOutput = <T::NOutput as ConnectSouth>::SRESULT;
    }
    pub trait Connects: ConnectNorth + ConnectEast + ConnectSouth + ConnectWest + Passes {
        const SELF: Self;
    }
    pub trait Combine<Rhs: Connects> {
        type COutput: Connects;
        const CRESULT: Self::COutput;
    }
    pub trait BlockGrid<const W: usize, const H: usize> {
        fn block<'a, const X: usize, const Y: usize>() -> &'a impl BlockTrait;
    }
    // impl<B: BlockTrait> BlockGrid<1, 1> for B {

    // }
    macro_rules! expand_combined {
        (Output;; $alter:ty) => {
            $alter
        };
        (Output; N; $alter:ty) => {
            <$alter as ConnectNorth>::NOutput
        };

        (Output; E; $alter:ty) => {
            <$alter as ConnectEast>::EOutput
        };

        (Output; S; $alter:ty) => {
            <$alter as ConnectSouth>::SOutput
        };

        (Output; W; $alter:ty) => {
            <$alter as ConnectWest>::WOutput
        };
        (Output; $init:ident $($rest:ident)+; $alter:expr) => {
            expand_combined!(
                Output;
                ($rest)+;
                expand_combined!(Output; $init; $alter)
            )
        };
    }
    macro_rules! impl_connect {
        (Char; North; $src:literal; $dst:literal) => {
            impl ConnectNorth for Char<$src> {
                type NOutput = Char<$dst>;
                const NRESULT: Self::NOutput = Char::<$dst>;
            }

        };
        (Char; East; $src:literal; $dst:literal) => {
            impl ConnectEast for Char<$src> {
                type EOutput = Char<$dst>;
                const ERESULT: Self::EOutput = Char::<$dst>;
            }

        };
        (Char; South; $src:literal; $dst:literal) => {
            impl ConnectSouth for Char<$src> {
                type SOutput = Char<$dst>;
                const SRESULT: Self::SOutput = Char::<$dst>;
            }

        };
        (Char; West; $src:literal; $dst:literal) => {
            impl ConnectWest for Char<$src> {
                type WOutput = Char<$dst>;
                const WRESULT: Self::WOutput = Char::<$dst>;
            }

        };

        (Char; Combine ($($dirs:ident)*); $src:literal) => {
            impl<Rhs: Connects> Combine<Rhs> for Char<$src> {
                type COutput = expand_combined!(Output; $($dirs)*; Rhs);
                const CRESULT: Self::COutput = <Self::COutput as Connects>::SELF;
            }
        };

        (Char; $src:literal; ($($dirs:ident)*); $nn:literal, $ee:literal, $ss:literal, $ww:literal) => {
            impl_connect!(Char; North; $src; $nn);
            impl_connect!(Char; East; $src; $ee);
            impl_connect!(Char; South; $src; $ss);
            impl_connect!(Char; West; $src; $ww);
            impl_connect!(Char; Combine ($($dirs)*); $src);
            impl Connects for Char<$src> {
                const SELF: Self = Char::<$src>;
            }
        }
    }

    impl_connect!(Char; ' '; (); '╵', '╶', '╷', '╴');

    impl_connect!(Char; '╵'; (N); '╵', '└', '│', '┘');
    impl_connect!(Char; '╶'; (E); '└', '╶', '┌', '─');
    impl_connect!(Char; '╷'; (S); '│', '┌', '╷', '┐');
    impl_connect!(Char; '╴'; (W); '┘', '─', '┐', '╴');

    impl<Rhs: Connects> Combine<Rhs> for Char<'└'> {
        type COutput = expand_combined!(Output;
            E;
            expand_combined!(Output;
            N;
            Rhs));
        const CRESULT: Self::COutput = <Self::COutput as Connects>::SELF;
    }
    // impl_connect!(Char; '└'; (N E); '└', '└', '├', '┴');
    impl_connect!(Char; '│'; (N S); '│', '├', '│', '┤');
    impl_connect!(Char; '┘'; (N W); '┘', '┴', '┤', '┘');

    impl_connect!(Char; '┌'; (E S); '├', '┌', '┌', '┬');
    impl_connect!(Char; '─'; (E W); '┴', '─', '┬', '─');

    impl_connect!(Char; '┐'; (S W);'┤', '┬', '┐', '┐');

    impl_connect!(Char; '├'; (N E S); '├', '├', '├', '┼');
    impl_connect!(Char; '┴'; (N E W); '├', '├', '┼', '├');
    impl_connect!(Char; '┤'; (N S W); '├', '┼', '├', '├');
    impl_connect!(Char; '┬'; (E S W); '┼', '├', '├', '├');

    impl_connect!(Char; '┼'; (N E S W); '┼', '┼', '┼', '┼');

    #[rustfmt::skip]
    impl<
        NW: Connects, NN: Connects, NE: Connects,
        WW: Connects, CC: Connects, EE: Connects,
        SW: Connects, SS: Connects, SE: Connects,
    > ConnectNorth for Track<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    > {
        type NOutput = Track<
            NW, NN::VOutput, NE,
            WW, CC::NOutput, EE,
            SW, SS,          SE,
        >;
        const NRESULT: Self::NOutput = strack();
    }
    #[rustfmt::skip]
    impl<
        NW: Connects, NN: Connects, NE: Connects,
        WW: Connects, CC: Connects, EE: Connects,
        SW: Connects, SS: Connects, SE: Connects,
    > ConnectEast for Track<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    > {
        type EOutput = Track<
            NW, NN,          NE,
            WW, CC::EOutput, EE::HOutput,
            SW, SS,          SE,
        >;
        const ERESULT: Self::EOutput = strack();
    }
    #[rustfmt::skip]
    impl<
        NW: Connects, NN: Connects, NE: Connects,
        WW: Connects, CC: Connects, EE: Connects,
        SW: Connects, SS: Connects, SE: Connects,
    > ConnectSouth for Track<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    > {
        type SOutput = Track<
            NW, NN,          NE,
            WW, CC::SOutput, EE,
            SW, SS::VOutput, SE,
        >;
        const SRESULT: Self::SOutput = strack();
    }
    #[rustfmt::skip]
    impl<
        NW: Connects, NN: Connects, NE: Connects,
        WW: Connects, CC: Connects, EE: Connects,
        SW: Connects, SS: Connects, SE: Connects,
    > ConnectWest for Track<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    > {
        type WOutput = Track<
            NW,          NN,          NE,
            WW::HOutput, CC::WOutput, EE,
            SW,          SS,          SE,
        >;
        const WRESULT: Self::WOutput = strack();
    }
    #[rustfmt::skip]
    impl<
        NW: Connects, NN: Connects, NE: Connects,
        WW: Connects, CC: Connects, EE: Connects,
        SW: Connects, SS: Connects, SE: Connects,
    > Connects for Track<
        NW, NN, NE,
        WW, CC, EE,
        SW, SS, SE,
    > {
        const SELF: Self = strack();
    }

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

    #[rustfmt::skip]
    macro_rules! def_block {
        ($name:ident, $($($c:literal)*);*) => {
            pub type $name = Block<$($(Char<$c>),*),*>;
            // impl<
            //     const NW: char, const NN: char, const NE: char,
            //     const WW: char, const CC: char, const EE: char,
            //     const SW: char, const SS: char, const SE: char,
            // > From<$name> for Block<NW, NN, NE, WW, CC, EE, SW, SS, SE>
            // {
            //     fn from(value: $name) -> Self {
            //         Self
            //     }
            // }
        };
        // impl<
        //     const NW: char, const NN: char, const NE: char,
        //     const WW: char, const CC: char, const EE: char,
        //     const SW: char, const SS: char, const SE: char,
        // > BlockTrait for $name:ident
        // {
        //     fn from(value: $name) -> Self {
        //         Self
        //     }
        // }
    }
    // pub struct NULL;
    def_block!(NULL,
        ' ' ' ' ' ';
        ' ' ' ' ' ';
        ' ' ' ' ' '
    );

    pub mod track {
        use super::{Block, Char, NULL};

        type OOOO = NULL;

        def_block!(OOOW,
            ' ' ' ' ' ';
            '─' '─' ' ';
            ' ' ' ' ' '
        );
        def_block!(OOSO,
            ' ' ' ' ' ';
            ' ' '│' ' ';
            ' ' '│' ' '
        );
        def_block!(OEOO,
            ' ' ' ' ' ';
            ' ' '─' '─';
            ' ' ' ' ' '
        );
        def_block!(NOOO,
            ' ' '│' ' ';
            ' ' '│' ' ';
            ' ' ' ' ' '
        );
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
