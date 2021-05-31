use crate::envs::Env;
use tch::{nn::VarStore, Tensor};

pub trait Policy<E: Env> {
    fn eval(&mut self, state: &Vec<f32>) -> (Vec<f32>, f32);
}

pub trait NNPolicy<E: Env> {
    fn new(vs: &VarStore) -> Self;
    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor);
}
