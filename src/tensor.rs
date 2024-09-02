use core::ops::{Deref, DerefMut};
use std::ops::Range;

pub type F = f32;
pub type Q = i8;

pub const GS: usize = 64;

#[derive(Clone, Copy)]
pub struct Tensor<'t, const N: usize> {
    pub shape: [usize; N],
    pub data: &'t [F],
}

#[derive(Clone, Copy)]
pub struct TensorQuant<'t, const N: usize> {
    pub shape: [usize; N],
    pub data: &'t [Q],
    pub scaling_factors: &'t [F],
}

pub struct TensorQuantMut<'t, const N: usize> {
    pub shape: [usize; N],
    pub data: &'t mut [Q],
    pub scaling_factors: &'t mut [F],
}

pub struct TensorMut<'t, const N: usize> {
    pub shape: [usize; N],
    pub data: &'t mut [F],
}

pub type Tensor1D<'t> = Tensor<'t, 1>;
pub type Tensor2D<'t> = Tensor<'t, 2>;

pub type TensorQuant1D<'t> = TensorQuant<'t, 1>;
pub type TensorQuant2D<'t> = TensorQuant<'t, 2>;

pub type TensorMut1D<'t> = TensorMut<'t, 1>;
pub type TensorMut2D<'t> = TensorMut<'t, 2>;

pub type TensorQuantMut1D<'t> = TensorQuantMut<'t, 1>;
pub type TensorQuantMut2D<'t> = TensorQuantMut<'t, 2>;

impl<'t, const N: usize> TensorMut<'t, N> {
    pub fn freeze(&'t self) -> Tensor<'t, N> {
        Tensor {
            shape: self.shape,
            data: &*self.data,
        }
    }

    pub fn slice_mut(&mut self, range: Range<usize>) -> TensorMut1D<'_> {
        let data = &mut self.data[range];
        TensorMut {
            shape: [data.len()],
            data,
        }
    }
}

impl<'t, const N: usize> TensorQuantMut<'t, N> {
    pub fn freeze(&'t self) -> TensorQuant<'t, N> {
        TensorQuant {
            shape: self.shape,
            data: &*self.data,
            scaling_factors: &*self.scaling_factors,
        }
    }
}

impl<'t, const N: usize> AsRef<[F]> for Tensor<'t, N> {
    fn as_ref(&self) -> &[F] {
        &self.data
    }
}

impl<'t, const N: usize> AsRef<[F]> for TensorMut<'t, N> {
    fn as_ref(&self) -> &[F] {
        &self.data
    }
}

impl<'t, const N: usize> AsMut<[F]> for TensorMut<'t, N> {
    fn as_mut(&mut self) -> &mut [F] {
        &mut self.data
    }
}

impl<'t, const N: usize> Deref for Tensor<'t, N> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'t, const N: usize> Deref for TensorMut<'t, N> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'t, const N: usize> DerefMut for TensorMut<'t, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

pub mod casts {
    use super::*;

    pub fn to_float(buf: &[u8]) -> &[F] {
        let (h, mid, t) = unsafe { buf.align_to::<F>() };
        assert!(h.is_empty() && t.is_empty());
        mid
    }

    pub fn to_float_mut(buf: &mut [u8]) -> &mut [F] {
        let (h, mid, t) = unsafe { buf.align_to_mut::<F>() };
        assert!(h.is_empty() && t.is_empty());
        mid
    }

    pub fn to_quant(buf: &[u8]) -> &[Q] {
        let (h, mid, t) = unsafe { buf.align_to::<Q>() };
        assert!(h.is_empty() && t.is_empty());
        mid
    }

    pub fn to_quant_mut(buf: &mut [u8]) -> &mut [Q] {
        let (h, mid, t) = unsafe { buf.align_to_mut::<Q>() };
        assert!(h.is_empty() && t.is_empty());
        mid
    }
}
