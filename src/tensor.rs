use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub, BitAnd, BitOr};
use cust::memory::{CopyDestination, DeviceBuffer, DeviceCopy};
use cust::util::SliceExt;
use crate::shape::{BroadcastShape, Shape};

struct CpuTensor<T, S: Shape> {
    data: Vec<T>,
    shape: S,
}

impl<T, S: Shape> CpuTensor<T, S> {
    fn zero(shape: S) -> Self
        where T: num_traits::Zero
    {
        let size = shape.size();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::zero());
        }

        Self { data, shape }
    }

    fn one(shape: S) -> Self
        where T: num_traits::One
    {
        let size = shape.size();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::one());
        }

        Self { data, shape }
    }

    fn of(shape: S, value: T) -> Self
        where T: Clone
    {
        let size = shape.size();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(value.clone());
        }

        Self { data, shape }
    }

    fn cuda(&self) -> CudaTensor<T, S>
        where T: DeviceCopy
    {
        // make sure the context and stream are set
        crate::CTX.with(|ctx| {
            crate::STREAM.with(|stream| {
                let dbuf = self.data.as_slice().as_dbuf().unwrap();

                CudaTensor {
                    data: dbuf,
                    shape: self.shape,
                }
            })
        })
    }

    fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

impl<T, S: Shape> Display for CpuTensor<T, S>
    where T: Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn fmt_inner<T: Display>(
            f: &mut std::fmt::Formatter<'_>,
            data: &[T],
            shape: &[usize],
            offset: usize,
        ) -> std::fmt::Result {
            if shape.len() == 1 {
                write!(f, "[")?;
                for i in 0..shape[0] {
                    write!(f, "{}", data[offset + i])?;
                    if i < shape[0] - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")?;
            } else {
                write!(f, "[")?;
                for i in 0..shape[0] {
                    fmt_inner(f, data, &shape[1..], offset + i * shape[1])?;
                    if i < shape[0] - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")?;
            }

            Ok(())
        }

        fmt_inner(f, self.data.as_slice(), self.shape.dimensions().as_slice(), 0)
    }
}

impl<T: DeviceCopy, S: Shape> Into<CudaTensor<T, S>> for CpuTensor<T, S> {
    fn into(self) -> CudaTensor<T, S> {
        self.cuda()
    }
}

struct CudaTensor<T: DeviceCopy, S: Shape> {
    data: DeviceBuffer<T>,
    shape: S,
}

impl<T: DeviceCopy, S: Shape> CudaTensor<T, S> {
    fn cpu(&self) -> CpuTensor<T, S> {
        let size = self.shape.size();
        let mut data = Vec::with_capacity(size);
        unsafe { data.set_len(size); };
        crate::STREAM.with(|stream| {
            stream.synchronize().unwrap();
            self.data.copy_to(&mut data).unwrap();

            CpuTensor {
                data,
                shape: self.shape,
            }
        })
    }
}

impl<T: DeviceCopy, S: Shape> Into<CpuTensor<T, S>> for CudaTensor<T, S> {
    fn into(self) -> CpuTensor<T, S> {
        self.cpu()
    }
}

macro_rules! impl_cuda_op {
    ($op_ty:ident, $fn_id:ident, $l_ty:ty, $r_ty:ty, $o_ty:ty, $cuda_op:expr) => {
        impl<
            SL: Shape,
            SR: Shape,
            SO: Shape,
        > $op_ty<&CudaTensor<$l_ty, SR>> for &CudaTensor<$r_ty, SL>
            where
                SL: BroadcastShape<SR, Output = SO>,
                generic_array::GenericArray<usize, <SO as Shape>::Dims>: Copy,
        {
            type Output = CudaTensor<$o_ty, SO>;

            fn $fn_id(self, rhs: &CudaTensor<$r_ty, SR>) -> Self::Output {
                let shape = self.shape.broadcast(rhs.shape);
                let size = shape.size();
                let mut out_buffer = unsafe { DeviceBuffer::uninitialized(size) }.unwrap();
                let o_strides = shape.strides();
                let a_dims = self.shape.dimensions();
                let b_dims = rhs.shape.dimensions();
                let mut a_strides = o_strides.clone();
                let mut b_strides = o_strides.clone();
                a_strides.fill(1);
                b_strides.fill(1);
                for i in 0..o_strides.len() {
                    if a_dims[i] == 1 && b_dims[i] > 1 {
                        a_strides[i] = 0;
                    }
                    if b_dims[i] == 1 && a_dims[i] > 1 {
                        b_strides[i] = 0;
                    }
                }

                let a_buffer = &self.data;
                let b_buffer = &rhs.data;

                crate::STREAM.with(|stream| {
                    crate::MODULE.with(|module| {
                        let func = module.get_function(format!(
                            $cuda_op,
                            o_strides.len()
                        )).unwrap();
                        let (_, block_size) = func.suggested_launch_configuration(
                            0, 0.into()
                        ).unwrap();
                        let grid_size = (size as u32 + block_size - 1) / block_size;
                        let a_strides = GenericArrayDeviceCopy::new(a_strides);
                        let b_strides = GenericArrayDeviceCopy::new(b_strides);
                        let o_strides = GenericArrayDeviceCopy::new(o_strides);

                        unsafe {
                            cust::launch!(
                            func<<<grid_size, block_size, 0, stream>>>(
                                a_buffer.as_device_ptr(),
                                a_buffer.len(),
                                a_strides,
                                b_buffer.as_device_ptr(),
                                b_buffer.len(),
                                b_strides,
                                out_buffer.as_device_ptr(),
                                out_buffer.len(),
                                o_strides,
                            )
                        ).unwrap();
                        }

                        CudaTensor {
                            data: out_buffer,
                            shape,
                        }
                    })
                })
            }
        }
    };
}


impl_cuda_op!(Add, add, f32, f32, f32, "add_f32_{}d");
impl_cuda_op!(Add, add, f64, f64, f64, "add_f64_{}d");
impl_cuda_op!(Sub, sub, f32, f32, f32, "sub_f32_{}d");
impl_cuda_op!(Sub, sub, f64, f64, f64, "sub_f64_{}d");
impl_cuda_op!(Mul, mul, f32, f32, f32, "mul_f32_{}d");
impl_cuda_op!(Mul, mul, f64, f64, f64, "mul_f64_{}d");
impl_cuda_op!(Div, div, f32, f32, f32, "div_f32_{}d");
impl_cuda_op!(Div, div, f64, f64, f64, "div_f64_{}d");

impl_cuda_op!(BitAnd, bitand, bool, bool, bool, "and_bool_{}d");
impl_cuda_op!(BitOr, bitor, bool, bool, bool, "or_bool_{}d");

#[derive(Clone, Copy)]
#[repr(transparent)]
struct GenericArrayDeviceCopy<T, N: generic_array::ArrayLength<T>>
    where
        T: Copy,
        generic_array::GenericArray<T, N>: Copy,
{
    data: generic_array::GenericArray<T, N>,
}
impl<T, N: generic_array::ArrayLength<T>> GenericArrayDeviceCopy<T, N>
    where
        T: Copy,
        generic_array::GenericArray<T, N>: Copy,
{
    fn new(data: generic_array::GenericArray<T, N>) -> Self {
        Self { data }
    }
}
unsafe impl<T, N: generic_array::ArrayLength<T>> DeviceCopy for GenericArrayDeviceCopy<T, N>
    where
        T: Copy,
        generic_array::GenericArray<T, N>: Copy,
{}

#[cfg(test)]
mod tests {
    use crate::shape::Cst;
    use super::*;

    #[test]
    fn test_add() {
        let cpu_tensor_a = CpuTensor::<f32, (Cst<typenum::U3>, Cst<typenum::U1>)>::one((Cst::new(), Cst::new()));
        let cpu_tensor_b = CpuTensor::<f32, (Cst<typenum::U1>, Cst<typenum::U3>)>::one((Cst::new(), Cst::new()));
        let cpu_tensor_a = cpu_tensor_a.cuda();
        let cpu_tensor_b = cpu_tensor_b.cuda();

        let cpu_tensor_o = &cpu_tensor_a + &cpu_tensor_b;
        let cpu_tensor_o = cpu_tensor_o.cpu();
        assert_eq!(cpu_tensor_o.as_slice(), &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_mul() {
        let cpu_tensor_a = CpuTensor::<f32, (Cst<typenum::U3>, Cst<typenum::U1>)>::one((Cst::new(), Cst::new()));
        let cpu_tensor_b = CpuTensor::<f32, (Cst<typenum::U1>, Cst<typenum::U3>)>::of((Cst::new(), Cst::new()), 3.0);
        let cpu_tensor_a = cpu_tensor_a.cuda();
        let cpu_tensor_b = cpu_tensor_b.cuda();

        let cpu_tensor_o = &cpu_tensor_a * &cpu_tensor_b;
        let cpu_tensor_o = cpu_tensor_o.cpu();
        assert_eq!(cpu_tensor_o.as_slice(), &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_div() {
        let cpu_tensor_a = CpuTensor::<f32, (Cst<typenum::U3>, Cst<typenum::U1>)>::of((Cst::new(), Cst::new()), 4.0);
        let cpu_tensor_b = CpuTensor::<f32, (Cst<typenum::U1>, Cst<typenum::U3>)>::of((Cst::new(), Cst::new()), 2.0);
        let cpu_tensor_a = cpu_tensor_a.cuda();
        let cpu_tensor_b = cpu_tensor_b.cuda();

        let cpu_tensor_o = &cpu_tensor_a / &cpu_tensor_b;
        let cpu_tensor_o = cpu_tensor_o.cpu();
        assert_eq!(cpu_tensor_o.as_slice(), &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_sub() {
        let cpu_tensor_a = CpuTensor::<f32, (Cst<typenum::U3>, Cst<typenum::U1>)>::one((Cst::new(), Cst::new()));
        let cpu_tensor_b = CpuTensor::<f32, (Cst<typenum::U1>, Cst<typenum::U3>)>::of((Cst::new(), Cst::new()), 3.0);
        let cpu_tensor_a = cpu_tensor_a.cuda();
        let cpu_tensor_b = cpu_tensor_b.cuda();

        let cpu_tensor_o = &cpu_tensor_a - &cpu_tensor_b;
        let cpu_tensor_o = cpu_tensor_o.cpu();
        assert_eq!(cpu_tensor_o.as_slice(), &[-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]);
    }
}