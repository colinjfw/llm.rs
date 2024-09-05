use core::ops::{Deref, DerefMut};

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
    pub sf: &'t [F],
}

pub struct TensorQuantMut<'t, const N: usize> {
    pub shape: [usize; N],
    pub data: &'t mut [Q],
    pub sf: &'t mut [F],
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

impl<'t, const N: usize> TensorMut<'t, N> {
    pub fn freeze(&'t self) -> Tensor<'t, N> {
        Tensor {
            shape: self.shape,
            data: &*self.data,
        }
    }
}

impl<'t, const N: usize> TensorQuantMut<'t, N> {
    pub fn freeze(&'t self) -> TensorQuant<'t, N> {
        TensorQuant {
            shape: self.shape,
            data: &*self.data,
            sf: &*self.sf,
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

impl<'t, const N: usize> From<TensorMut<'t, N>> for Tensor<'t, N> {
    fn from(tensor: TensorMut<'t, N>) -> Self {
        Tensor {
            shape: tensor.shape,
            data: tensor.data,
        }
    }
}

#[inline(always)]
pub fn quantize(out: &mut TensorQuantMut1D, from: Tensor1D) {
    const Q_MAX: f32 = 127.0;

    let num_groups = from.len() / GS;

    for group in 0..num_groups {
        // Find the max absolute value in the current group
        let mut wmax = 0.0;
        for i in 0..GS {
            let val = from[group * GS + i].abs();
            if val > wmax {
                wmax = val;
            }
        }

        // Calculate and write the scaling factor
        let scale = wmax / Q_MAX;
        out.sf[group] = scale;

        // Calculate and write the quantized values
        for i in 0..GS {
            let quant_value = from[group * GS + i] / scale; // scale
            let quantized = (quant_value.round() as i8).clamp(-128, 127); // round and clamp
            out.data[group * GS + i] = quantized;
        }
    }
}

#[inline(always)]
pub fn dequantize<const N: usize>(out: &mut TensorMut<N>, from: TensorQuant<N>) {
    for i in 0..from.data.len() {
        out[i] = (from.data[i] as F) * from.sf[i / GS];
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
