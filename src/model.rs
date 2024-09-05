//! Inference for Llama-2 Transformer model in pure Rust

use core::mem::MaybeUninit;
use std::{
    io::{self, Write},
    time::{Duration, Instant},
};

use crate::{parallel::parallel, tensor::*, tokenizer::Tokenizer};

const MAX_LAYERS: usize = 32;
const MAX_HEADS: usize = 32;
const MAX_SEQ_LEN: usize = 4096;
const MAX_HEAD_SIZE: usize = MAX_SEQ_LEN / MAX_LAYERS;

#[derive(Debug, Clone, Copy)]
pub struct ModelParams {
    pub dim: usize,        // transformer dimension
    pub hidden_dim: usize, // for ffn layers
    pub n_layers: usize,   // number of layers
    pub n_heads: usize,    // number of query heads
    pub n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize,    // max sequence length
}

struct Buffers<'m> {
    x: TensorMut1D<'m>,                  // (dim)
    xb: TensorMut1D<'m>,                 // (dim)
    xq: TensorQuantMut1D<'m>,            // (dim)
    hb: TensorMut1D<'m>,                 // (hidden_dim)
    hb2: TensorMut1D<'m>,                // (hidden_dim)
    hq: TensorQuantMut1D<'m>,            // (hidden_dim)
    q: TensorMut1D<'m>,                  // (dim)
    k: TensorMut1D<'m>,                  // (dim)
    v: TensorMut1D<'m>,                  // (dim)
    att: TensorMut2D<'m>,                // (n_heads, seq_len)
    logits: TensorMut1D<'m>,             // (vocab_size)
    key_cache: TensorMut1D<'m>,          // (layer * seq_len * dim)
    val_cache: TensorMut1D<'m>,          // (layer * seq_len * dim)
    token_embedding_table: Tensor2D<'m>, // (vocab_size, dim)
    rms_final_weight: Tensor1D<'m>,      // (dim,)
}

#[derive(Clone, Copy)]
struct Layer<'l> {
    rms_att_weight: Tensor1D<'l>, // (dim)
    rms_ffn_weight: Tensor1D<'l>, // (dim)
    // matmuls
    /// (dim, heads * head_size)
    wq: TensorQuant2D<'l>,
    /// (dim, n_kv_heads * head_size)
    wk: TensorQuant2D<'l>,
    /// (dim, n_kv_heads * head_size)
    wv: TensorQuant2D<'l>,
    /// (n_heads * head_size, dim)
    wo: TensorQuant2D<'l>,
    // weights for ffn
    w1: TensorQuant2D<'l>, // (hidden_dim, dim)
    w2: TensorQuant2D<'l>, // (dim, hidden_dim)
    w3: TensorQuant2D<'l>, // (hidden_dim, dim)
}

struct Weights<'m> {
    token_embedding_table: TensorQuant2D<'m>, // (vocab_size, dim)
    rms_final_weight: TensorQuant1D<'m>,      // (dim,)
    wcls: TensorQuant2D<'m>,                  // (vocab_size, dim)
    layers: [MaybeUninit<Layer<'m>>; MAX_LAYERS],
}

pub struct Model<'m> {
    tokenizer: Tokenizer<'m>,
    params: ModelParams,
    weights: Weights<'m>,
    buffers: Buffers<'m>,
}

// Neural net blocks; the dynamics of the Transformer

/// rmsnorm(x(s), w(s)) -> o(s)
#[inline(always)]
fn rmsnorm(o: &mut TensorMut1D, x: Tensor1D, w: Tensor1D) {
    debug_assert_eq!(x.shape, w.shape);
    debug_assert_eq!(x.shape, o.shape);

    let mut ss = 0.0f32;
    for v in x.iter().copied() {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();
    for j in 0..w.len() {
        o[j] = w[j] * (ss * x[j]);
    }
}

#[inline(always)]
fn rmsnorm_self(o: &mut TensorMut1D, w: Tensor1D) {
    debug_assert_eq!(w.shape, o.shape);

    let mut ss = 0.0f32;
    for v in o.iter().copied() {
        ss += v * v;
    }
    ss /= o.len() as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();
    for j in 0..w.len() {
        o[j] = w[j] * (ss * o[j]);
    }
}

#[inline(always)]
fn softmax(x: &mut [f32]) {
    let size = x.len();
    let mut max_val = x[0];
    for i in 1..size {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    let mut sum = 0.0f32;
    for i in 0..size {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }
    for i in 0..size {
        x[i] /= sum;
    }
}

/// W (d,n) @ x (n,) -> xout (d,)
#[inline(always)]
fn matmul(out: &mut TensorMut1D, w: TensorQuant2D, x: TensorQuant1D) {
    let (d, n) = (w.shape[0], w.shape[1]);
    debug_assert_eq!(x.shape[0], n);

    for i in 0..d {
        let mut val = 0.0;
        let mut ival = 0i32;
        let in_idx = i * n;

        // Perform the matmul in groups of GS
        let mut j = 0;
        while j <= n - GS {
            for k in 0..GS {
                ival += (x.data[j + k] as i32) * (w.data[in_idx + j + k] as i32);
            }
            val += (ival as f32) * w.sf[(in_idx + j) / GS] * x.sf[j / GS];
            ival = 0;
            j += GS;
        }

        out[i] = val;
    }
}

// RoPE relative positional encoding: complex-valued rotate q and k in each head
#[inline(always)]
fn rope(
    q: &mut TensorMut1D,
    k: &mut TensorMut1D,
    pos: usize,
    dim: usize,
    kv_dim: usize,
    head_size: usize,
) {
    for i in (0..dim).step_by(2) {
        let head_dim = i % head_size;
        let freq = 1.0f32 / 10000.0f32.powf(head_dim as f32 / head_size as f32);
        let val = pos as f32 * freq;
        let (fcr, fci) = (val.cos(), val.sin());

        let (v0, v1) = (q[i], q[i + 1]);
        q[i] = v0 * fcr - v1 * fci;
        q[i + 1] = v0 * fci + v1 * fcr;

        if i < kv_dim {
            let (v0, v1) = (k[i], k[i + 1]);
            k[i] = v0 * fcr - v1 * fci;
            k[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

#[inline(always)]
fn multihead_attention(
    key_cache: Tensor1D,
    val_cache: Tensor1D,
    q: Tensor1D,
    xb: &mut TensorMut1D,
    pos: usize,
    loff: usize,
    n_heads: usize,
    kv_dim: usize,
    kv_mul: usize,
    head_size: usize,
    seq_len: usize,
) {
    // on the stack
    let mut att = TensorMut1D {
        shape: [seq_len],
        data: &mut [0.0; MAX_SEQ_LEN][..seq_len],
    };

    for h in 0..n_heads {
        // get the query vector for this head
        let q = &q[h * head_size..(h + 1) * head_size];

        // iterate over all timesteps, including the current one
        for t in 0..=pos {
            // get the key vector for this head and at this timestep
            let k = &key_cache[loff + t * kv_dim + (h / kv_mul) * head_size..];
            // calculate the attention score as the dot product of q and k
            let score: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();
            // save the score to the attention buffer
            att[t] = score / (head_size as f32).sqrt();
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(&mut att[0..pos + 1]);

        // weighted sum of the values, store back into xb
        let xb = &mut xb[h * head_size..(h + 1) * head_size];
        xb.fill(0.0);
        for t in 0..=pos {
            // get the value vector for this head and at this timestep
            let v = &val_cache[loff + t * kv_dim + (h / kv_mul) * head_size..];
            // get the attention weight for this timestep
            let a = att[t];
            // accumulate the weighted value into xb
            for i in 0..head_size {
                xb[i] += a * v[i];
            }
        }
    }
}

// SwiGLU non-linearity
#[inline(always)]
fn swiglu(hb: &mut TensorMut1D, hb2: &mut TensorMut1D, hidden_dim: usize) {
    for i in 0..hidden_dim {
        let mut val = hb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= 1.0f32 / (1.0f32 + (-val).exp());
        // elementwise multiply with w3(x)
        val *= hb2[i];
        hb[i] = val;
    }
}

#[inline(always)]
fn sample(logits: Tensor1D) -> usize {
    let mut max_i = 0;
    let mut max_p = logits[0];
    for (i, p) in logits.iter().copied().enumerate() {
        if p > max_p {
            max_i = i;
            max_p = p;
        }
    }
    max_i
}

fn generate(model: &mut Model, start: usize, steps: usize) {
    let mut pos = 0;
    let mut token = start;

    let mut stdout = io::stdout().lock();
    let start = Instant::now();

    while pos < steps {
        forward(model, token, pos);

        let next = sample(model.buffers.logits.freeze());
        if next == Tokenizer::BOS {
            return;
        }

        let output = model.tokenizer.decode(token, next);
        stdout
            .write(output.as_bytes())
            .expect("error writing to stdout");
        stdout.flush().expect("error flushing");

        token = next;
        pos += 1;
    }

    let elapsed = start.elapsed();
    let elapsed_per_token = Duration::from_micros(elapsed.as_micros() as u64 / steps as u64);
    println!("\n\ngenerated {steps} tokens in {elapsed:?}, {elapsed_per_token:?} per token");
}

fn forward(model: &mut Model, token: usize, pos: usize) {
    let ModelParams {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        seq_len,
        ..
    } = model.params;
    let kv_dim = (dim * n_kv_heads) / n_heads;
    let kv_mul = n_heads / n_kv_heads;
    let head_size = dim / n_heads;

    let Buffers {
        x,
        xb,
        xq,
        hb,
        hb2,
        hq,
        q,
        k,
        v,
        logits,
        key_cache,
        val_cache,
        token_embedding_table,
        rms_final_weight,
        ..
    } = &mut model.buffers;
    let wcls = model.weights.wcls;

    let content_row = &token_embedding_table[token * dim..(token + 1) * dim];
    x.copy_from_slice(content_row);

    for l in 0..n_layers {
        let layer = unsafe { model.weights.layers[l].assume_init() };

        // attention rmsnorm
        rmsnorm(xb, x.freeze(), layer.rms_att_weight);

        // qkv matmuls for this position
        quantize(xq, xb.freeze());
        matmul(q, layer.wq, xq.freeze());
        matmul(k, layer.wk, xq.freeze());
        matmul(v, layer.wv, xq.freeze());

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        rope(q, k, pos, dim, kv_dim, head_size);

        // save key,value at this time step (pos) to our kv cache
        let loff = l * seq_len * kv_dim;
        key_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim].copy_from_slice(&k);
        val_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim].copy_from_slice(&v);

        // multihead attention. iterate over all heads
        multihead_attention(
            key_cache.freeze(),
            val_cache.freeze(),
            q.freeze(),
            xb,
            // att,
            pos,
            loff,
            n_heads,
            kv_dim,
            kv_mul,
            head_size,
            seq_len,
        );

        // final matmul to get the output of the attention
        quantize(xq, xb.freeze());
        matmul(xb, layer.wo, xq.freeze());

        // residual connection back into x
        for i in 0..dim {
            x[i] += xb[i];
        }

        // ffn rmsnorm
        rmsnorm(xb, x.freeze(), layer.rms_ffn_weight);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(xq, xb.freeze());
        matmul(hb, layer.w1, xq.freeze());
        matmul(hb2, layer.w3, xq.freeze());

        // SwiGLU non-linearity
        swiglu(hb, hb2, hidden_dim);

        // final matmul to get the output of the ffn
        quantize(hq, hb.freeze());
        matmul(xb, layer.w2, hq.freeze());

        // residual connection
        for i in 0..dim {
            x[i] += xb[i];
        }
    }

    // final rmsnorm
    rmsnorm_self(x, *rms_final_weight);

    // classifier into logits
    quantize(xq, x.freeze());
    matmul(logits, wcls, xq.freeze());
}

impl ModelParams {
    pub fn buffer_size(&self) -> usize {
        let ModelParams {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            vocab_size,
            seq_len,
            ..
        } = *self;

        let size_f = (dim * 6)
            + (hidden_dim * 2)
            + (n_heads * seq_len)
            + vocab_size
            + ((n_layers * seq_len * dim) * 2)
            + (vocab_size * dim);
        let size_q =
            dim + ((dim / 64) * size_of::<F>()) + hidden_dim + ((hidden_dim / 64) * size_of::<F>());
        (size_f * size_of::<F>()) + size_q
    }
}

impl<'m> Model<'m> {
    pub fn new(
        params: ModelParams,
        tokenizer: Tokenizer<'m>,
        mut weights: &'m [u8],
        mut alloc: &'m mut [u8],
    ) -> Self {
        let ModelParams {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        } = params;
        let head_size = dim / n_heads;

        macro_rules! tensor {
            ($shape:expr) => {{
                let len = $shape.iter().copied().product::<usize>() * size_of::<F>();
                let (data, next) = weights.split_at(len);
                weights = next;
                Tensor {
                    shape: $shape,
                    data: casts::to_float(data),
                }
            }};
        }

        macro_rules! tensor_mut {
            ($shape:expr) => {{
                let len = $shape.iter().copied().product::<usize>() * size_of::<F>();
                let (data, next) = alloc.split_at_mut(len);
                alloc = next;
                TensorMut {
                    shape: $shape,
                    data: casts::to_float_mut(data),
                }
            }};
        }

        macro_rules! tensor_quant {
            ($shape:expr) => {{
                let data_len = $shape.iter().copied().product::<usize>();
                let sf_len = (data_len / GS) * size_of::<F>();
                let len = data_len + sf_len;
                let (all, next) = weights.split_at(len);
                let (data, sf) = all.split_at(data_len);
                weights = next;
                TensorQuant {
                    shape: $shape,
                    data: casts::to_quant(data),
                    sf: casts::to_float(sf),
                }
            }};
        }

        macro_rules! tensor_quant_mut {
            ($shape:expr) => {{
                let data_len = $shape.iter().copied().product::<usize>();
                let sf_len = (data_len / GS) * size_of::<F>();
                let len = data_len + sf_len;
                let (all, next) = alloc.split_at_mut(len);
                let (data, sf) = all.split_at_mut(data_len);
                alloc = next;
                TensorQuantMut {
                    shape: $shape,
                    data: casts::to_quant_mut(data),
                    sf: casts::to_float_mut(sf),
                }
            }};
        }

        let mut layers = [const { MaybeUninit::uninit() }; MAX_LAYERS];
        let token_embedding_table = tensor_quant!([vocab_size, dim]);
        for i in 0..n_layers {
            let layer = Layer {
                rms_att_weight: tensor!([dim]),
                rms_ffn_weight: tensor!([dim]),
                wq: tensor_quant!([dim, n_heads * head_size]),
                wk: tensor_quant!([dim, n_kv_heads * head_size]),
                wv: tensor_quant!([dim, n_kv_heads * head_size]),
                wo: tensor_quant!([n_heads * head_size, dim]),
                w1: tensor_quant!([hidden_dim, dim]),
                w2: tensor_quant!([dim, hidden_dim]),
                w3: tensor_quant!([hidden_dim, dim]),
            };
            layers[i].write(layer);
        }
        let rms_final_weight = tensor_quant!([dim]);
        let wcls = tensor_quant!([vocab_size, dim]);

        debug_assert!(weights.is_empty(), "leftover weights: {}", weights.len());
        let weights = Weights {
            token_embedding_table,
            rms_final_weight,
            wcls,
            layers,
        };

        let mut rms_final_weight = tensor_mut!([dim]);
        dequantize(&mut rms_final_weight, weights.rms_final_weight);
        let mut token_embedding_table = tensor_mut!([vocab_size, dim]);
        dequantize(&mut token_embedding_table, weights.token_embedding_table);

        let buffers = Buffers {
            x: tensor_mut!([dim]),
            xb: tensor_mut!([dim]),
            xq: tensor_quant_mut!([dim]),
            hb: tensor_mut!([hidden_dim]),
            hb2: tensor_mut!([hidden_dim]),
            hq: tensor_quant_mut!([hidden_dim]),
            q: tensor_mut!([dim]),
            k: tensor_mut!([dim]),
            v: tensor_mut!([dim]),
            att: tensor_mut!([n_heads, seq_len]),
            logits: tensor_mut!([vocab_size]),
            key_cache: tensor_mut!([n_layers * seq_len * dim]),
            val_cache: tensor_mut!([n_layers * seq_len * dim]),
            token_embedding_table: token_embedding_table.into(),
            rms_final_weight: rms_final_weight.into(),
        };
        debug_assert!(alloc.is_empty(), "leftover alloc: {}", alloc.len());

        Self {
            params,
            weights,
            buffers,
            tokenizer,
        }
    }

    pub fn generate(&mut self) {
        generate(self, 463, 10)
    }
}
