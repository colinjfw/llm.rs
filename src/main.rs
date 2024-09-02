use std::{env::args, fs, os::fd::AsRawFd, ptr, slice};

mod model;
mod tensor;
mod tokenizer;

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
        "memory mapped {}GiB from weights file",
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
        model::ModelParams {
            dim: params[1] as _,
            hidden_dim: params[2] as _,
            n_layers: params[3] as _,
            n_heads: params[4] as _,
            n_kv_heads: params[5] as _,
            vocab_size: params[6] as _,
            seq_len: params[7] as _,
        }
    };

    let mut alloc = vec![0; params.buffer_size()];
    println!(
        "allocated {}GiB buffers",
        alloc.len() / (1024 * 1024 * 1024)
    );

    let tokenizer = tokenizer::Tokenizer {};

    let mut model = model::Model::new(params, tokenizer, &weights[header_size..], &mut alloc);
    println!("model built: {params:#?}");

    model.generate("tell me a story");
}
