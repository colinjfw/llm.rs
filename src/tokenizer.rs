use std::io;

use crate::model::ModelParams;

pub struct Tokenizer {}

impl Tokenizer {
    pub const BOS: usize = 1;

    pub fn new(params: &ModelParams) -> Self {
        Self {}
    }

    pub fn encode<'s>(&self, s: &'s str) -> Encode<'s> {
        Encode { i: 0, str: s }
    }

    pub fn decode(&self, prev: usize, next: usize, f: &mut impl io::Write) {
        todo!()
    }
}

pub struct Encode<'s> {
    i: usize,
    str: &'s str,
}

impl<'s> Iterator for Encode<'s> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
