use crate::connect4::Connect4;
use slimnn::{Activation, Linear, ReLU};
use synthesis::prelude::*;
use tch::{self, nn, Tensor};

pub struct Connect4Net {
    l_1: nn::Linear,
    l_2: nn::Linear,
    l_3: nn::Linear,
    l_4: nn::Linear,
}

impl NNPolicy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn new(vs: &nn::VarStore) -> Self {
        let root = &vs.root();
        let state_dims = Connect4::DIMS;
        assert!(state_dims.len() == 4);
        assert!(&state_dims == &[1, 1, 7, 9]);
        Self {
            l_1: nn::linear(root / "l_1", 63, 128, Default::default()),
            l_2: nn::linear(root / "l_2", 128, 96, Default::default()),
            l_3: nn::linear(root / "l_3", 96, 64, Default::default()),
            l_4: nn::linear(root / "l_4", 64, 10, Default::default()),
            // TODO add additional layer from 64 -> 48 -> 10
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs
            .flat_view()
            .apply(&self.l_1)
            .relu()
            .apply(&self.l_2)
            .relu()
            .apply(&self.l_3)
            .relu()
            .apply(&self.l_4);
        let mut ts = xs.split_with_sizes(&[9, 1], -1);
        let value = ts.pop().unwrap();
        let logits = ts.pop().unwrap();
        (logits, value)
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let xs = env.features();
        let t = tensor(&xs, Connect4::DIMS, tch::Kind::Float);
        let (logits, value) = self.forward(&t);
        let mut policy = [0.0f32; Connect4::MAX_NUM_ACTIONS];
        logits.copy_data(&mut policy, Connect4::MAX_NUM_ACTIONS);
        let value = f32::from(&value).clamp(-1.0, 1.0);
        (policy, value)
    }
}

#[derive(Default)]
pub struct SlimC4Net {
    l_1: Linear<63, 48>,
    l_2: Linear<48, 32>,
    l_3: Linear<32, 10>,
}

impl SlimC4Net {
    fn forward(&self, x: &[[[f32; 9]; 7]; 1]) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let x: [f32; 63] = unsafe { std::mem::transmute(*x) };

        let x = self.l_1.forward(&x);
        let x = ReLU.apply_1d(&x);
        let x = self.l_2.forward(&x);
        let x = ReLU.apply_1d(&x);
        let x = self.l_3.forward(&x);

        let mut logits = [0.0; 9];
        logits.copy_from_slice(&x[..Connect4::MAX_NUM_ACTIONS]);
        let value = x[Connect4::MAX_NUM_ACTIONS].clamp(-1.0, 1.0);

        (logits, value)
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for SlimC4Net {
    fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        self.forward(&env.features())
    }
}
