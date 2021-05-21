use rand::rngs::StdRng;
use tch::Tensor;

pub trait Env {
    type PlayerId: Eq + Clone + Copy + std::fmt::Debug;
    type Action: Eq + Clone + Copy + std::fmt::Debug;
    type ActionIterator: Iterator<Item = Self::Action>;

    fn new() -> Self;
    fn player(&self) -> Self::PlayerId;
    fn is_over(&self) -> bool;
    fn reward(&self, color: Self::PlayerId) -> f32;
    fn iter_actions(&self) -> Self::ActionIterator;
    fn num_actions(&self) -> u8;
    fn get_random_action(&self, rng: &mut StdRng) -> Self::Action;
    fn step(&mut self, action: &Self::Action) -> bool;
    fn state(&self) -> Tensor;
}
