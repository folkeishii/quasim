macro_rules! complex {
    ($real: expr, $imag: expr) => {
        Complex::new($real, $imag)
    };
}
pub(super) use complex;

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
