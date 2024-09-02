use crate::model::ModelParams;

pub struct Tokenizer<'t> {
    vocab: &'t [&'t str],
}

impl<'t> Tokenizer<'t> {
    pub const BOS: usize = 1;

    pub fn new(params: &ModelParams, vocab: &'t mut [&'t str], tokens: &'t [u8]) -> Self {
        let mut pos = 0;
        for i in 0..params.vocab_size {
            let len = tokens[pos] as usize;
            let str = std::str::from_utf8(&tokens[pos + 1..pos + 1 + len]).unwrap();
            vocab[i] = str;
            pos += len + 1;
        }

        Self { vocab }
    }

    pub fn decode(&self, prev: usize, next: usize) -> &str {
        let mut piece = self.vocab[next];
        if prev == Self::BOS && piece == " " {
            piece = self.vocab[next + 1];
        }
        piece
    }
}
