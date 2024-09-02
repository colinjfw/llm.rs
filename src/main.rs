// Inference for Llama-2 Transformer model in pure Rust

use std::{env::args, fs, io::Read, mem::MaybeUninit, os::fd::AsRawFd, ptr, slice};

use tensor::*;

mod tensor;

const L: usize = 32;

#[derive(Debug, Clone, Copy)]
struct ModelParams {
    dim: usize,        // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize,   // number of layers
    n_heads: usize,    // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize,    // max sequence length
}

struct Buffers<'m> {
    x: TensorMut1D<'m>,         // (dim)
    xb: TensorMut1D<'m>,        // (dim)
    xb2: TensorMut1D<'m>,       // (dim)
    xq: TensorQuantMut1D<'m>,   // (dim)
    hb: TensorMut1D<'m>,        // (hidden_dim)
    hb2: TensorMut1D<'m>,       // (hidden_dim)
    hq: TensorQuantMut1D<'m>,   // (hidden_dim)
    q: TensorMut1D<'m>,         // (dim)
    att: TensorMut2D<'m>,       // (n_heads, seq_len)
    logits: TensorMut1D<'m>,    // (vocab_size)
    key_cache: TensorMut1D<'m>, // (layer * seq_len * dim)
    val_cache: TensorMut1D<'m>, // (layer * seq_len * dim)
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
    layers: [MaybeUninit<Layer<'m>>; L],
}

struct Model<'m> {
    params: ModelParams,
    weights: Weights<'m>,
    buffers: Buffers<'m>,
}

// Neural net blocks; the dynamics of the Transformer

/// rmsnorm(x(s), w(s)) -> o(s)
#[inline(always)]
fn rmsnorm(o: &mut TensorMut1D, x: Tensor1D, w: Tensor1D) {
    assert_eq!(x.shape, w.shape);
    assert_eq!(x.shape, o.shape);

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
    assert_eq!(w.shape, o.shape);

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

#[inline(always)]
fn quantize(out: &mut TensorQuantMut1D, from: Tensor1D) {}

/// W (d,n) @ x (n,) -> xout (d,)
#[inline(always)]
fn matmul(out: &mut TensorMut1D, w: TensorQuant2D, x: TensorQuant1D) {
    // // W (d,n) @ x (n,) -> xout (d,)
    // let (d, n) = (w.shape[0], w.shape[1]);
    // assert_eq!(x.shape[0], n);

    // for i in 0..d {
    //     let mut val = 0.0;
    //     for j in 0..n {
    //         val += w[i * n + j] * x[j];
    //     }
    //     out[i] = val;
    // }
}

fn forward(model: &mut Model, token: usize, pos: usize) {
    let p = model.params;
    let w = &model.weights;

    let x = &mut model.buffers.x;
    let xb = &mut model.buffers.xb;
    let xb2 = &mut model.buffers.xb2;
    let xq = &mut model.buffers.xq;
    let hq = &mut model.buffers.hq;
    let q = &mut model.buffers.q;
    let att = &mut model.buffers.att;
    let hb = &mut model.buffers.hb;
    let hb2 = &mut model.buffers.hb2;
    let key_cache = &mut model.buffers.key_cache;
    let val_cache = &mut model.buffers.val_cache;
    let logits = &mut model.buffers.logits;

    let dim = p.dim;
    let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    let kv_mul = p.n_heads / p.n_kv_heads;
    let hidden_dim = p.hidden_dim;
    let head_size = dim / p.n_heads;

    // let content_row = &w.token_embedding_table[token * dim..(token + 1) * dim];
    // x.copy_from_slice(content_row);

    for l in 0..p.n_layers {
        let layer = unsafe { w.layers[l].assume_init() };

        // attention rmsnorm
        rmsnorm(xb, x.freeze(), layer.rms_att_weight);

        // key and value point to the kv cache
        let loff = l * p.seq_len * kv_dim;
        {
            let mut k = key_cache.slice_mut(loff + pos * kv_dim..loff + (pos + 1) * kv_dim);
            let mut v = val_cache.slice_mut(loff + pos * kv_dim..loff + (pos + 1) * kv_dim);

            let k = &mut k;
            let v = &mut v;

            // qkv matmuls for this position
            quantize(xq, xb.freeze());
            matmul(q, layer.wq, xq.freeze());
            matmul(k, layer.wk, xq.freeze());
            matmul(v, layer.wv, xq.freeze());

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
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

        // multihead attention. iterate over all heads
        for h in 0..p.n_heads {
            // get the query vector for this head
            let q = &q[h * head_size..(h + 1) * head_size];
            // attention scores for this head
            let att = &mut att[h * p.seq_len..(h + 1) * p.seq_len];
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
            for t in 0..=pos {
                // get the value vector for this head and at this timestep
                let v = &val_cache[loff + t * kv_dim + (h / kv_mul) * head_size..];
                // get the attention weight for this timestep
                let a = att[t];
                // accumulate the weighted value into xb
                for (xb_i, &v_i) in xb.iter_mut().zip(v.iter()) {
                    *xb_i += a * v_i;
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(xq, xb.freeze());
        matmul(xb2, layer.wo, xq.freeze());

        // residual connection back into x
        for i in 0..dim {
            x[i] += xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(xb, x.freeze(), layer.rms_ffn_weight);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(xq, xb.freeze());
        matmul(hb, layer.w1, xq.freeze());
        matmul(hb2, layer.w3, xq.freeze());

        // SwiGLU non-linearity
        for i in 0..hidden_dim {
            let val = hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            let val = val * (1.0f32 / (1.0f32 + (-val).exp()));
            // elementwise multiply with w3(x)
            hb[i] = val * hb2[i];
        }

        // final matmul to get the output of the ffn
        quantize(hq, hb.freeze());
        matmul(xb, layer.w2, hq.freeze());

        // residual connection
        for i in 0..dim {
            x[i] += xb[i];
        }
    }

    // final rmsnorm
    // rmsnorm_self(x, w.rms_final_weight);

    // classifier into logits
    quantize(xq, x.freeze());
    matmul(logits, w.wcls, xq.freeze());
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

        let size_f = (dim * 4)
            + (hidden_dim * 2)
            + (n_heads * seq_len)
            + vocab_size
            + ((n_layers * seq_len * dim) * 2);
        let size_q =
            dim + ((dim / 64) * size_of::<F>()) + hidden_dim + ((hidden_dim / 64) * size_of::<F>());
        (size_f * size_of::<F>()) + size_q
    }
}

impl<'m> Model<'m> {
    fn new(params: ModelParams, mut weights: &'m [u8], mut alloc: &'m mut [u8]) -> Self {
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
                    scaling_factors: casts::to_float(sf),
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
                    scaling_factors: casts::to_float_mut(sf),
                }
            }};
        }

        let mut layers = [const { MaybeUninit::uninit() }; L];
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

        assert!(weights.is_empty(), "leftover weights: {}", weights.len());
        let weights = Weights {
            token_embedding_table,
            rms_final_weight,
            wcls,
            layers,
        };

        let buffers = Buffers {
            x: tensor_mut!([dim]),
            xb: tensor_mut!([dim]),
            xb2: tensor_mut!([dim]),
            xq: tensor_quant_mut!([dim]),
            hb: tensor_mut!([hidden_dim]),
            hb2: tensor_mut!([hidden_dim]),
            hq: tensor_quant_mut!([hidden_dim]),
            q: tensor_mut!([dim]),
            att: tensor_mut!([n_heads, seq_len]),
            logits: tensor_mut!([vocab_size]),
            key_cache: tensor_mut!([n_layers * seq_len * dim]),
            val_cache: tensor_mut!([n_layers * seq_len * dim]),
        };
        assert!(alloc.is_empty(), "leftover alloc: {}", alloc.len());

        Self {
            params,
            weights,
            buffers,
        }
    }
}

pub fn main() {
    let filepath = args().nth(1).expect("must provide filepath");
    println!("reading {filepath}...");

    let file = fs::File::open(filepath).expect("could not open file");
    let metadata = file.metadata().expect("could not read file");

    let weights = unsafe {
        let len = metadata.len() as usize;
        let ptr = libc::mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        );
        let ptr = ptr as *const u8;
        slice::from_raw_parts(ptr, len)
    };
    println!(
        "mmaped {}GiB from weights file",
        weights.len() / (1024 * 1024 * 1024)
    );

    let header_size = 8 * size_of::<u32>();
    let params = {
        let header = &weights[..header_size];
        let params = {
            let (h, buf, t) = unsafe { header.align_to::<u32>() };
            assert!(h.is_empty() && t.is_empty());
            buf
        };

        assert_eq!(params[0], 12440); // version
        ModelParams {
            dim: params[1] as _,
            hidden_dim: params[2] as _,
            n_layers: params[3] as _,
            n_heads: params[4] as _,
            n_kv_heads: params[5] as _,
            vocab_size: params[6] as _,
            seq_len: params[7] as _,
        }
    };
    println!("read model header");

    let mut alloc = vec![0; params.buffer_size()];
    println!(
        "allocated {}GiB buffers",
        alloc.len() / (1024 * 1024 * 1024)
    );

    let mut model = Model::new(params, &weights[header_size..], &mut alloc);
    println!("model built: {params:#?}");

    forward(&mut model, 0, 0);
}
