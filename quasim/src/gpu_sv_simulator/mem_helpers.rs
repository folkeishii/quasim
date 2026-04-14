use nalgebra::Complex;

pub fn bytes_from_complex<T>(data: &[Complex<T>]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

pub unsafe fn complex_from_bytes<T>(bytes: &[u8]) -> &[Complex<T>] {
    assert_eq!(bytes.len() % size_of::<Complex<T>>(), 0);
    assert_eq!(bytes.as_ptr().align_offset(align_of::<Complex<T>>()), 0);

    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().cast::<Complex<T>>(),
            bytes.len() / size_of::<Complex<T>>(),
        )
    }
}

pub fn bytes_from_complex_mut<T>(data: &mut [Complex<T>]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            data.len() * size_of::<Complex<T>>(),
        )
    }
}

pub unsafe fn complex_from_bytes_mut<T>(bytes: &mut [u8]) -> &mut [Complex<T>] {
    assert_eq!(bytes.len() % size_of::<Complex<T>>(), 0);
    assert_eq!(bytes.as_ptr().align_offset(align_of::<Complex<T>>()), 0);

    unsafe {
        std::slice::from_raw_parts_mut(
            bytes.as_mut_ptr() as *mut Complex<T>,
            bytes.len() / size_of::<Complex<T>>(),
        )
    }
}
