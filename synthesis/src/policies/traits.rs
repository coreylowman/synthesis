use crate::game::Game;
use tch::{nn::VarStore, Tensor};

pub trait Policy<G: Game<N>, const N: usize> {
    fn eval(&mut self, game: &G) -> ([f32; N], [f32; 3]);
}

pub trait NNPolicy<G: Game<N>, const N: usize> {
    fn new(vs: &VarStore) -> Self;
    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor);
}
