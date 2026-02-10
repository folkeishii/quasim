use std::{cmp::Ordering, f32::consts::FRAC_1_SQRT_2};

use nalgebra::{Complex, Const, DVector, Dyn, SVector};
use quasim::{AT_00, AT_11, Gate2x2, ID, SQsystem, ST, STATE_0, TP, TPV, cart, cmp_elements,};

extern crate quasim;
fn main() {
    divan::main();
}

#[divan::bench()]
fn dyn_hadamard_cnot_entanglement() {
    let hadamard = Gate2x2::new(
        cart!(FRAC_1_SQRT_2),
        cart!(FRAC_1_SQRT_2),
        cart!(FRAC_1_SQRT_2),
        cart!(-FRAC_1_SQRT_2),
    );
    let x = Gate2x2::new(cart!(0.0), cart!(1.0), cart!(1.0), cart!(0.0));

    let tp0 = TP::from(vec![hadamard, ID, ID]);
    let tp1 = TP::from(vec![AT_00, ID, ID]);
    let tp2 = TP::from(vec![AT_11, ID, x]);

    let st0: ST<Dyn> = tp0.into();
    let st1 = tp1 + tp2;

    let mst = st1 * st0;

    let state = TPV::from(vec![STATE_0, STATE_0, STATE_0]);
    assert_eq!(
        cmp_elements(
            &(mst * state).eval(),
            &DVector::from_row_slice(&[
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
            ]),
            0.000001,
        ),
        Some(Ordering::Equal)
    );
}

#[divan::bench()]
fn stack_hadamard_cnot_entanglement() {
    let hadamard = Gate2x2::new(
        FRAC_1_SQRT_2.into(),
        FRAC_1_SQRT_2.into(),
        FRAC_1_SQRT_2.into(),
        (-FRAC_1_SQRT_2).into(),
    );
    let x = Gate2x2::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

    let tp0 = TP::from([hadamard, ID, ID]);
    let tp1 = TP::from([AT_00, ID, ID]);
    let tp2 = TP::from([AT_11, ID, x]);

    let st0: ST<Const<3>> = tp0.into();
    let st1 = tp1 + tp2;

    let mst = st1 * st0;

    let state = TPV::from([STATE_0, STATE_0, STATE_0]);

    assert_eq!(
        cmp_elements(
            &(mst * state).eval(),
            &SVector::from_row_slice(&[
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
            ]),
            0.000001,
        ),
        Some(Ordering::Equal)
    );
}

#[divan::bench()]
fn dyn_hadamard_cnot_entanglement_final_matrix() {
    let hadamard = Gate2x2::new(
        FRAC_1_SQRT_2.into(),
        FRAC_1_SQRT_2.into(),
        FRAC_1_SQRT_2.into(),
        (-FRAC_1_SQRT_2).into(),
    );
    let x = Gate2x2::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

    let tp0 = TP::from(vec![hadamard, ID, ID]);
    let tp1 = TP::from(vec![AT_00, ID, ID]);
    let tp2 = TP::from(vec![AT_11, ID, x]);

    let st0: ST<Dyn> = tp0.into();
    let st1 = tp1 + tp2;

    let mst = st1 * st0;

    let final_matrix = mst.eval();
    let state = TPV::from(vec![STATE_0, STATE_0, STATE_0]).eval();

    assert_eq!(
        cmp_elements(
            &(final_matrix * state),
            &DVector::from_row_slice(&[
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
            ]),
            0.000001,
        ),
        Some(Ordering::Equal)
    );
}

#[divan::bench()]
fn stack_hadamard_cnot_entanglement_final_matrix() {
    let hadamard = Gate2x2::new(
        FRAC_1_SQRT_2.into(),
        FRAC_1_SQRT_2.into(),
        FRAC_1_SQRT_2.into(),
        (-FRAC_1_SQRT_2).into(),
    );
    let x = Gate2x2::new(0.0.into(), 1.0.into(), 1.0.into(), 0.0.into());

    let tp0 = TP::from([hadamard, ID, ID]);
    let tp1 = TP::from([AT_00, ID, ID]);
    let tp2 = TP::from([AT_11, ID, x]);

    let st0: ST<Const<3>> = tp0.into();
    let st1 = tp1 + tp2;

    let mst = st1 * st0;

    let final_matrix = mst.eval();
    let state = TPV::from([STATE_0, STATE_0, STATE_0]).eval();

    assert_eq!(
        cmp_elements(
            &(final_matrix * state),
            &SVector::from_row_slice(&[
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(0.0),
                cart!(FRAC_1_SQRT_2),
                cart!(0.0),
                cart!(0.0),
            ]),
            0.000001,
        ),
        Some(Ordering::Equal)
    );
}

const fn gate_at(bit_count: usize, index: usize, i: usize) -> Gate2x2 {
    if i == bit_count - 1 {
        return ID;
    }
    if (index & (1 << i)) > 0 { AT_11 } else { AT_00 }
}
#[divan::bench(
    types = [Const<2>, Const<4>, Const<6>, Const<8>, Const<10>, ],
    args = [1, 2, 3, 4]
)]
fn stack_n_qubit_m_control_identity<QS: SQsystem>(len: usize) {
    let circuit_len = len;
    let system = QS::system();

    let tp0 =
        TP::<QS>::from_array(system.init_gate_storage(&|i| gate_at(system.bit_count(), 0, i)));
    let tp1 =
        TP::<QS>::from_array(system.init_gate_storage(&|i| gate_at(system.bit_count(), 1, i)));
    let mut st = tp0 + tp1;
    for index in 2..(1 << (system.bit_count() - 1)) {
        st += TP::<QS>::from_array(
            system.init_gate_storage(&|i| gate_at(system.bit_count(), index, i)),
        )
        .into();
    }
    let mut mst = st.clone() * st.clone();
    for _ in 2..circuit_len {
        mst *= st.clone().into();
    }

    let state = TPV::<QS>::from_array(system.init_state_storage(&|_| STATE_0));

    assert_eq!(
        cmp_elements(
            &(mst * state).eval(),
            &system.init_system_state(&|i| if i == 0 { cart!(1.0) } else { cart!(0.0) }),
            0.000001,
        ),
        Some(Ordering::Equal)
    );
}
