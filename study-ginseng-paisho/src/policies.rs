use tch::nn::VarStore;
use tch::Tensor;
use synthesis::prelude::{NNPolicy, Policy};
use crate::ginseng::Ginseng;
use synthesis::game::Game;

pub struct GinsengNet;

impl Policy<Ginseng, { Ginseng::MAX_NUM_ACTIONS }> for GinsengNet {
    fn eval(&mut self, game: &Ginseng) -> ([f32; Ginseng::MAX_NUM_ACTIONS], [f32; 3]) {
        todo!()
    }
}

impl NNPolicy<Ginseng, {Ginseng::MAX_NUM_ACTIONS}> for GinsengNet {
    fn new(vs: &VarStore) -> Self {
        todo!()
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        todo!()
    }
}