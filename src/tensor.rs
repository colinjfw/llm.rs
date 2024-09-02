use core::ops::{Deref, DerefMut};
use std::ops::Range;

pub type F = f32;
pub type Q = i8;

const GS: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct Tensor<'t, const N: usize> {
    pub shape: [usize; N],
    pub data: &'t [F],
}

// #[derive(Debug, Clone, Copy)]
// pub struct TensorQuant<'t, const N: usize> {
//     pub shape: [usize; N],
//     pub data: &'t [Q],
//     pub scaling_factors: &'t [F],
// }

#[derive(Debug)]
pub struct TensorMut<'t, const N: usize> {
    pub shape: [usize; N],
    pub data: &'t mut [F],
}

pub type Tensor1D<'t> = Tensor<'t, 1>;
pub type Tensor2D<'t> = Tensor<'t, 2>;
// pub type Tensor3D<'t> = Tensor<'t, 3>;

pub type TensorMut1D<'t> = TensorMut<'t, 1>;
pub type TensorMut2D<'t> = TensorMut<'t, 2>;
// pub type TensorMut3D<'t> = TensorMut<'t, 3>;

impl<'t, const N: usize> TensorMut<'t, N> {
    pub fn freeze(&'t self) -> Tensor<'t, N> {
        Tensor {
            shape: self.shape,
            data: &*self.data,
        }
    }

    pub fn slice_mut(&mut self, range: Range<usize>) -> TensorMut1D<'_> {
        let data = &mut self.data[range];
        TensorMut { shape: [data.len()], data }
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
