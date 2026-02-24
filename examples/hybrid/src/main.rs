use quasim::{c, cconst, circuit::Circuit, register::RegisterFileRef};

fn main() {
    let _circ1_1 = Circuit::new(2)
        // add registers
        // measure
        .jump_if(
            |file: &mut RegisterFileRef<usize>| file["x"] * file["y"] <= 10,
            "end".into(),
        )
        .x(1)
        .fn_classic(|file: &mut RegisterFileRef<usize>| {
            file["y"] += 2;
        })
        // label end
        ;
    let _circ1_2 = Circuit::new(2)
        // add registers
        // measure
        .jump_if(
            cmpp,
            "end".into(),
        )
        .x(1)
        .fn_classic(foo)
        // label end
        ;
    let _circ2 = Circuit::new(2)
        // add registers
        // measure
        .jump_if(
            (c!("x") * c!("y")).le(cconst!(10)),
            "end".into(),
        )
        .x(1)
        .fn_classic(c!("y").assign(c!("y") + cconst!(2)))
        // label end
        ;
}

fn cmpp(file: &mut RegisterFileRef<usize>) -> bool {
    file["x"] * file["y"] <= 10
}

fn foo(file: &mut RegisterFileRef<usize>) -> () {
    file["y"] += 2
}
