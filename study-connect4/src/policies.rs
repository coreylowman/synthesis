use crate::connect4::Connect4;
use slimnn::{Activation, Conv2d, Linear, ReLU};
use synthesis::prelude::*;
use tch::{
    self,
    nn::{self, ConvConfig},
    Tensor,
};

pub struct Connect4Net {
    c_1: nn::Conv2D,
    l_1: nn::Linear,
    l_2: nn::Linear,
}

impl NNPolicy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn new(vs: &nn::VarStore) -> Self {
        let root = &vs.root();
        let state_dims = Connect4::DIMS;
        assert!(state_dims.len() == 4);
        assert!(&state_dims == &[1, 3, 7, 9]);
        Self {
            c_1: nn::conv2d(
                root / "c_1",
                3,
                5,
                3,
                ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ),
            l_1: nn::linear(root / "l_1", 60, 48, Default::default()),
            l_2: nn::linear(root / "l_2", 48, 10, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs
            .apply(&self.c_1)
            .relu()
            .flat_view()
            .apply(&self.l_1)
            .relu()
            .apply(&self.l_2);
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
    c_1: Conv2d<3, 5, 3, 0, 0, 2>,
    l_1: Linear<60, 48>,
    l_2: Linear<48, 10>,
}

impl SlimC4Net {
    fn forward(&self, x: &[[[f32; 9]; 7]; 3]) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let x = self.c_1.forward::<9, 7, 4, 3>(x);
        let x = ReLU.apply_3d(&x);

        let x: [f32; 60] = unsafe { std::mem::transmute(x) };

        let x = self.l_1.forward(&x);
        let x = ReLU.apply_1d(&x);
        let x = self.l_2.forward(&x);

        let mut logits = [0.0; 9];
        logits.copy_from_slice(&x[..9]);
        let value = x[9].clamp(-1.0, 1.0);

        (logits, value)
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for SlimC4Net {
    fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        self.forward(&env.features())
    }
}
