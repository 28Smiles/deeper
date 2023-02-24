#![allow(improper_ctypes_definitions)]
#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;

macro_rules! impl_op {
    ($fn_name:ident, $l_ty:ty, $r_ty:ty, $o_ty:ty, $dim:ty, $op:expr) => {
        #[kernel]
        #[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
        pub unsafe fn $fn_name(
            a: &[$l_ty],
            a_strides: generic_array::GenericArray<usize, $dim>,
            b: &[$r_ty],
            b_strides: generic_array::GenericArray<usize, $dim>,
            o: *mut $o_ty,
            o_size: usize,
            o_strides: generic_array::GenericArray<usize, $dim>,
        ) {
            let o = core::slice::from_raw_parts_mut(o, o_size);
            let idx = thread::index_1d() as usize;
            apply_op_broadcast(
                a,
                a_strides,
                b,
                b_strides,
                o,
                o_strides,
                $op,
                idx,
            );
        }
    };
}

impl_op!(add_f32_1d, f32, f32, f32, generic_array::typenum::U1, |a, b| a + b);
impl_op!(add_f32_2d, f32, f32, f32, generic_array::typenum::U2, |a, b| a + b);
impl_op!(add_f32_3d, f32, f32, f32, generic_array::typenum::U3, |a, b| a + b);
impl_op!(add_f32_4d, f32, f32, f32, generic_array::typenum::U4, |a, b| a + b);
impl_op!(add_f32_5d, f32, f32, f32, generic_array::typenum::U5, |a, b| a + b);
impl_op!(add_f32_6d, f32, f32, f32, generic_array::typenum::U6, |a, b| a + b);
impl_op!(mul_f32_1d, f32, f32, f32, generic_array::typenum::U1, |a, b| a * b);
impl_op!(mul_f32_2d, f32, f32, f32, generic_array::typenum::U2, |a, b| a * b);
impl_op!(mul_f32_3d, f32, f32, f32, generic_array::typenum::U3, |a, b| a * b);
impl_op!(mul_f32_4d, f32, f32, f32, generic_array::typenum::U4, |a, b| a * b);
impl_op!(mul_f32_5d, f32, f32, f32, generic_array::typenum::U5, |a, b| a * b);
impl_op!(mul_f32_6d, f32, f32, f32, generic_array::typenum::U6, |a, b| a * b);
impl_op!(sub_f32_1d, f32, f32, f32, generic_array::typenum::U1, |a, b| a - b);
impl_op!(sub_f32_2d, f32, f32, f32, generic_array::typenum::U2, |a, b| a - b);
impl_op!(sub_f32_3d, f32, f32, f32, generic_array::typenum::U3, |a, b| a - b);
impl_op!(sub_f32_4d, f32, f32, f32, generic_array::typenum::U4, |a, b| a - b);
impl_op!(sub_f32_5d, f32, f32, f32, generic_array::typenum::U5, |a, b| a - b);
impl_op!(sub_f32_6d, f32, f32, f32, generic_array::typenum::U6, |a, b| a - b);
impl_op!(div_f32_1d, f32, f32, f32, generic_array::typenum::U1, |a, b| a / b);
impl_op!(div_f32_2d, f32, f32, f32, generic_array::typenum::U2, |a, b| a / b);
impl_op!(div_f32_3d, f32, f32, f32, generic_array::typenum::U3, |a, b| a / b);
impl_op!(div_f32_4d, f32, f32, f32, generic_array::typenum::U4, |a, b| a / b);
impl_op!(div_f32_5d, f32, f32, f32, generic_array::typenum::U5, |a, b| a / b);
impl_op!(div_f32_6d, f32, f32, f32, generic_array::typenum::U6, |a, b| a / b);
impl_op!(eq_f32_1d, f32, f32, bool, generic_array::typenum::U1, |a, b| a == b);
impl_op!(eq_f32_2d, f32, f32, bool, generic_array::typenum::U2, |a, b| a == b);
impl_op!(eq_f32_3d, f32, f32, bool, generic_array::typenum::U3, |a, b| a == b);
impl_op!(eq_f32_4d, f32, f32, bool, generic_array::typenum::U4, |a, b| a == b);
impl_op!(eq_f32_5d, f32, f32, bool, generic_array::typenum::U5, |a, b| a == b);
impl_op!(eq_f32_6d, f32, f32, bool, generic_array::typenum::U6, |a, b| a == b);

impl_op!(add_f64_1d, f64, f64, f64, generic_array::typenum::U1, |a, b| a + b);
impl_op!(add_f64_2d, f64, f64, f64, generic_array::typenum::U2, |a, b| a + b);
impl_op!(add_f64_3d, f64, f64, f64, generic_array::typenum::U3, |a, b| a + b);
impl_op!(add_f64_4d, f64, f64, f64, generic_array::typenum::U4, |a, b| a + b);
impl_op!(add_f64_5d, f64, f64, f64, generic_array::typenum::U5, |a, b| a + b);
impl_op!(add_f64_6d, f64, f64, f64, generic_array::typenum::U6, |a, b| a + b);
impl_op!(mul_f64_1d, f64, f64, f64, generic_array::typenum::U1, |a, b| a * b);
impl_op!(mul_f64_2d, f64, f64, f64, generic_array::typenum::U2, |a, b| a * b);
impl_op!(mul_f64_3d, f64, f64, f64, generic_array::typenum::U3, |a, b| a * b);
impl_op!(mul_f64_4d, f64, f64, f64, generic_array::typenum::U4, |a, b| a * b);
impl_op!(mul_f64_5d, f64, f64, f64, generic_array::typenum::U5, |a, b| a * b);
impl_op!(mul_f64_6d, f64, f64, f64, generic_array::typenum::U6, |a, b| a * b);
impl_op!(sub_f64_1d, f64, f64, f64, generic_array::typenum::U1, |a, b| a - b);
impl_op!(sub_f64_2d, f64, f64, f64, generic_array::typenum::U2, |a, b| a - b);
impl_op!(sub_f64_3d, f64, f64, f64, generic_array::typenum::U3, |a, b| a - b);
impl_op!(sub_f64_4d, f64, f64, f64, generic_array::typenum::U4, |a, b| a - b);
impl_op!(sub_f64_5d, f64, f64, f64, generic_array::typenum::U5, |a, b| a - b);
impl_op!(sub_f64_6d, f64, f64, f64, generic_array::typenum::U6, |a, b| a - b);
impl_op!(div_f64_1d, f64, f64, f64, generic_array::typenum::U1, |a, b| a / b);
impl_op!(div_f64_2d, f64, f64, f64, generic_array::typenum::U2, |a, b| a / b);
impl_op!(div_f64_3d, f64, f64, f64, generic_array::typenum::U3, |a, b| a / b);
impl_op!(div_f64_4d, f64, f64, f64, generic_array::typenum::U4, |a, b| a / b);
impl_op!(div_f64_5d, f64, f64, f64, generic_array::typenum::U5, |a, b| a / b);
impl_op!(div_f64_6d, f64, f64, f64, generic_array::typenum::U6, |a, b| a / b);
impl_op!(eq_f64_1d, f64, f64, bool, generic_array::typenum::U1, |a, b| a == b);
impl_op!(eq_f64_2d, f64, f64, bool, generic_array::typenum::U2, |a, b| a == b);
impl_op!(eq_f64_3d, f64, f64, bool, generic_array::typenum::U3, |a, b| a == b);
impl_op!(eq_f64_4d, f64, f64, bool, generic_array::typenum::U4, |a, b| a == b);
impl_op!(eq_f64_5d, f64, f64, bool, generic_array::typenum::U5, |a, b| a == b);
impl_op!(eq_f64_6d, f64, f64, bool, generic_array::typenum::U6, |a, b| a == b);

impl_op!(eq_bool_1d, bool, bool, bool, generic_array::typenum::U1, |a, b| a == b);
impl_op!(eq_bool_2d, bool, bool, bool, generic_array::typenum::U2, |a, b| a == b);
impl_op!(eq_bool_3d, bool, bool, bool, generic_array::typenum::U3, |a, b| a == b);
impl_op!(eq_bool_4d, bool, bool, bool, generic_array::typenum::U4, |a, b| a == b);
impl_op!(eq_bool_5d, bool, bool, bool, generic_array::typenum::U5, |a, b| a == b);
impl_op!(eq_bool_6d, bool, bool, bool, generic_array::typenum::U6, |a, b| a == b);
impl_op!(and_bool_1d, bool, bool, bool, generic_array::typenum::U1, |a, b| a & b);
impl_op!(and_bool_2d, bool, bool, bool, generic_array::typenum::U2, |a, b| a & b);
impl_op!(and_bool_3d, bool, bool, bool, generic_array::typenum::U3, |a, b| a & b);
impl_op!(and_bool_4d, bool, bool, bool, generic_array::typenum::U4, |a, b| a & b);
impl_op!(and_bool_5d, bool, bool, bool, generic_array::typenum::U5, |a, b| a & b);
impl_op!(and_bool_6d, bool, bool, bool, generic_array::typenum::U6, |a, b| a & b);
impl_op!(or_bool_1d, bool, bool, bool, generic_array::typenum::U1, |a, b| a | b);
impl_op!(or_bool_2d, bool, bool, bool, generic_array::typenum::U2, |a, b| a | b);
impl_op!(or_bool_3d, bool, bool, bool, generic_array::typenum::U3, |a, b| a | b);
impl_op!(or_bool_4d, bool, bool, bool, generic_array::typenum::U4, |a, b| a | b);
impl_op!(or_bool_5d, bool, bool, bool, generic_array::typenum::U5, |a, b| a | b);
impl_op!(or_bool_6d, bool, bool, bool, generic_array::typenum::U6, |a, b| a | b);

#[inline(always)]
fn apply_op_broadcast<
    D: Into<[usize; DIMS]>,
    L: Copy,
    R: Copy,
    O: Copy,
    F: Fn(L, R) -> O,
    const DIMS: usize,
>(
    a: &[L],
    a_strides: D,
    b: &[R],
    b_strides: D,
    o: &mut [O],
    o_strides: D,
    op: F,
    idx: usize,
) {
    let a_strides = a_strides.into();
    let b_strides = b_strides.into();
    let o_strides = o_strides.into();
    if idx < o.len() {
        let mut a_idx = 0;
        let mut b_idx = 0;
        let mut o_idx = idx;
        for i in 0..DIMS {
            let a_stride = a_strides[i];
            let b_stride = b_strides[i];
            let o_stride = o_strides[i];
            let o_idx_dim = o_idx / o_stride;
            a_idx += o_idx_dim * a_stride;
            b_idx += o_idx_dim * b_stride;
            o_idx -= o_idx_dim * o_stride;
        }
        o[idx] = op(a[a_idx], b[b_idx]);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let mut o = [0.0; 9];
        let a_strides = [0, 1];
        let b_strides = [1, 0];
        let o_strides = [3, 1];
        for idx in 0..100 {
            apply_op_broadcast(
                &a,
                a_strides,
                &b,
                b_strides,
                &mut o,
                o_strides,
                |a, b| a + b,
                idx,
            );
        }

        std::assert_eq!(o, [
            2.0, 3.0, 4.0,
            3.0, 4.0, 5.0,
            4.0, 5.0, 6.0,
        ]);
    }
}
