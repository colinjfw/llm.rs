use std::{env::args, fs, io::Read, os::fd::AsRawFd, ptr, slice};

use std::alloc::System;

#[global_allocator]
static A: System = System;

mod model;
mod tensor;
mod tokenizer;
mod parallel;

pub fn main() {
    let mut args = args();

    let weight_file = args.nth(1).expect("must provide weight file");
    let token_file = args.next().expect("must provide token file");

    println!("reading weights from {weight_file}...");
    let weights = unsafe {
        let file = fs::File::open(weight_file).expect("could not open file");
        let metadata = file.metadata().expect("could not read file");

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

    println!("reading tokens from {token_file}...");
    let mut vocab = vec![""; params.vocab_size];
    let mut tokens = Vec::new();
    {
        let mut file = fs::File::open(token_file).expect("could not open file");
        file.read_to_end(&mut tokens).expect("could not read file");
    }
    println!("allocated {}KiB tokens", tokens.len() / (1024));

    let tokenizer = tokenizer::Tokenizer::new(&params, &mut vocab, &tokens);
    let mut model = model::Model::new(params, tokenizer, &weights[header_size..], &mut alloc);
    println!("model built: {params:#?}");

    model.generate();
}
