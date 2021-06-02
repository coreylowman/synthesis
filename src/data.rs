use std::ffi::c_void;
use tch::{Kind, Tensor};
use torch_sys::at_tensor_of_data;
use crate::envs::Env;

pub struct BatchRandSampler<'a> {
    inds: Tensor,

    x: &'a Tensor,
    y: &'a Tensor,
    z: &'a Tensor,

    size: i64,
    batch_size: i64,
    index: i64,
    drop_last: bool,
}

impl<'a> BatchRandSampler<'a> {
    pub fn new(
        x: &'a Tensor,
        y: &'a Tensor,
        z: &'a Tensor,
        batch_size: i64,
        drop_last: bool,
    ) -> Self {
        let n = x.size()[0];
        Self {
            inds: Tensor::randperm(n, tch::kind::INT64_CPU),
            x,
            y,
            z,
            size: n,
            batch_size,
            index: 0,
            drop_last,
        }
    }
}

impl<'a> Iterator for BatchRandSampler<'a> {
    type Item = (Tensor, Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let next_index = (self.index + self.batch_size).min(self.size);
        if self.index >= self.size
            || (self.drop_last && (next_index - self.index) < self.batch_size)
        {
            return None;
        }

        let batch_inds = self
            .inds
            .narrow(0, self.index as i64, (next_index - self.index) as i64);
        self.index = next_index;

        let item = (
            self.x.index_select(0, &batch_inds),
            self.y.index_select(0, &batch_inds),
            self.z.index_select(0, &batch_inds),
        );
        Some(item)
    }
}

pub fn tensor<T>(data: &[T], dims: &[i64], kind: tch::Kind) -> Tensor {
    let dsize = kind.elt_size_in_bytes();
    let dtype = match kind {
        Kind::Uint8 => 0,
        Kind::Int8 => 1,
        Kind::Int16 => 2,
        Kind::Int => 3,
        Kind::Int64 => 4,
        Kind::Half => 5,
        Kind::Float => 6,
        Kind::Double => 7,
        Kind::ComplexHalf => 8,
        Kind::ComplexFloat => 9,
        Kind::ComplexDouble => 10,
        Kind::Bool => 11,
        Kind::QInt8 => 12,
        Kind::QUInt8 => 13,
        Kind::QInt32 => 14,
        Kind::BFloat16 => 15,
    };
    let data = data.as_ptr() as *const c_void;
    let ndims = dims.len();
    let dims = dims.as_ptr();
    unsafe { Tensor::from_ptr(at_tensor_of_data(data, dims, ndims, dsize, dtype)) }
}

pub struct ReplayBuffer<E: Env<N>, const N: usize> {
    capacity: usize,
    pub states: Vec<E::State>,
    pub pis: Vec<[f32; N]>,
    pub vs: Vec<f32>,
}

impl<E: Env<N>, const N: usize> ReplayBuffer<E, N> {
    pub fn new(n: usize) -> Self {
        Self {
            capacity: n,
            states: Vec::with_capacity(n),
            pis: Vec::with_capacity(n),
            vs: Vec::with_capacity(n),
        }
    }

    pub fn add(&mut self, state: &E::State, pi: &[f32; N], v: f32) {
        self.states.push(state.clone());
        self.pis.push(pi.clone());
        self.vs.push(v);
    }

    pub fn make_room(&mut self, n: usize) {
        if self.vs.len() + n > self.capacity {
            let num_to_drop = self.vs.len() + n - self.capacity;
            drop(self.states.drain(0..(num_to_drop)));
            drop(self.pis.drain(0..(num_to_drop)));
            drop(self.vs.drain(0..(num_to_drop)));
        }
        assert!(self.vs.len() + n <= self.capacity);
    }
}
