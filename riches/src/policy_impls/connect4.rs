use crate::envs::Connect4;
use ragz::prelude::*;
use slimnn::{Activation, Conv2d, Linear, ReLU, Softmax};
use tch::{
    self,
    nn::{self, ConvConfigND},
    Tensor,
};

pub struct Connect4Net {
    c_1: nn::Conv2D,
    p_1: nn::Linear,
    p_2: nn::Linear,
    v_1: nn::Linear,
    v_2: nn::Linear,
}

impl NNPolicy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn new(vs: &nn::VarStore) -> Self {
        let root = &vs.root();
        let state_dims = Connect4::get_state_dims();
        assert!(state_dims.len() == 4);
        assert!(state_dims[0] == 1);
        Self {
            c_1: nn::conv(
                root / "c_1",
                2,
                5,
                [3, 3],
                ConvConfigND {
                    stride: [3, 3],
                    padding: [1, 0],
                    ..Default::default()
                },
            ),
            p_1: nn::linear(root / "p_1", 5 * 3 * 3, 32, Default::default()),
            p_2: nn::linear(
                root / "p_2",
                32,
                Connect4::MAX_NUM_ACTIONS as i64,
                Default::default(),
            ),
            v_1: nn::linear(root / "v_1", 5 * 3 * 3, 32, Default::default()),
            v_2: nn::linear(root / "v_2", 32, 1, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs.apply(&self.c_1).relu().flat_view();
        (
            xs.apply(&self.p_1).relu().apply(&self.p_2),
            xs.apply(&self.v_1).relu().apply(&self.v_2).tanh(),
        )
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn eval(
        &mut self,
        xs: &<Connect4 as Env<{ Connect4::MAX_NUM_ACTIONS }>>::State,
    ) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let t = tensor(xs, &Connect4::get_state_dims(), tch::Kind::Bool).to_kind(tch::Kind::Float);
        let (logits, value) = self.forward(&t);
        let mut policy = [0.0f32; Connect4::MAX_NUM_ACTIONS];
        logits
            .softmax(-1, tch::Kind::Float)
            .copy_data(&mut policy, Connect4::MAX_NUM_ACTIONS);
        let value = f32::from(&value);
        (policy, value)
    }
}

#[derive(Default)]
pub struct SlimC4Net {
    c_1: Conv2d<2, 5, 3, 1, 0, 3>,
    p_1: Linear<{ 5 * 3 * 3 }, 32>,
    p_2: Linear<32, { Connect4::MAX_NUM_ACTIONS }>,
    v_1: Linear<{ 5 * 3 * 3 }, 32>,
    v_2: Linear<32, 1>,
}

impl SlimC4Net {
    fn forward(&self, x: &[[[f32; 9]; 7]; 2]) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let x = self.c_1.forward::<9, 7, 3, 3>(x);
        let x = ReLU.apply_3d(&x);

        let x: [f32; 5 * 3 * 3] = unsafe { std::mem::transmute(x) };

        let px = self.p_1.forward(&x);
        let px = ReLU.apply_1d(&px);
        let logits = self.p_2.forward(&px);

        let vx = self.v_1.forward(&x);
        let vx = ReLU.apply_1d(&vx);
        let value = self.v_2.forward(&vx)[0].tanh();

        (logits, value)
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for SlimC4Net {
    fn eval(
        &mut self,
        xs: &<Connect4 as Env<{ Connect4::MAX_NUM_ACTIONS }>>::State,
    ) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let mut x = [[[0.0; 9]; 7]; 2];
        for i in 0..2 {
            for j in 0..7 {
                for k in 0..9 {
                    x[i][j][k] = if xs[i][j][k] { 1.0 } else { 0.0 };
                }
            }
        }
        let (logits, value) = self.forward(&x);
        let policy = Softmax.apply_1d(&logits);
        (policy, value)
    }
}
