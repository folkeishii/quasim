macro_rules! cartesian {
    ($real: expr, $imag: expr) => {
        Complex::new($real, $imag)
    };
}
pub(super) use cartesian;

macro_rules! polar {
    ($r: expr, $theta: expr) => {
        Complex::from_polar($r, $theta)
    };
}
pub(super) use polar;

macro_rules! real {
    ($real: expr) => {
        Complex::new($real, 0.0)
    };
}
pub(super) use real;

macro_rules! imag {
    ($imag: expr) => {
        Complex::new(0.0, $imag)
    };
}
pub(super) use imag;

macro_rules! cexp {
    ($exp: expr) => {
        Complex::exp($exp)
    };
}
pub(super) use cexp;
