use crate::envs::Connect4;
use ragz::prelude::*;
use slimnn::{Activation, Conv2d, Linear, ReLU, Softmax};
use tch::{
    self,
    nn::{self, ConvConfig},
    Tensor,
};

fn to_float<const W: usize, const H: usize, const C: usize>(
    x_bool: &[[[bool; W]; H]; C],
) -> [[[f32; W]; H]; C] {
    let mut x_f32 = [[[0.0; W]; H]; C];
    for i in 0..C {
        let bool_i = &x_bool[i];
        let f32_i = &mut x_f32[i];
        for j in 0..H {
            let bool_j = &bool_i[j];
            let f32_j = &mut f32_i[j];
            for k in 0..W {
                if bool_j[k] {
                    f32_j[k] = 1.0;
                }
            }
        }
    }
    x_f32
}

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
        assert!(&state_dims == &[1, 2, 7, 9]);
        Self {
            c_1: nn::conv2d(
                root / "c_1",
                2,
                5,
                3,
                ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ),
            p_1: nn::linear(root / "p_1", 60, 32, Default::default()),
            p_2: nn::linear(
                root / "p_2",
                32,
                Connect4::MAX_NUM_ACTIONS as i64,
                Default::default(),
            ),
            v_1: nn::linear(root / "v_1", 60, 32, Default::default()),
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
    fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let state = env.state();
        let xs = to_float(&state);
        let t = tensor(&xs, &[1, 2, 7, 9], tch::Kind::Float);
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
    c_1: Conv2d<2, 5, 3, 0, 0, 2>,
    p_1: Linear<60, 32>,
    p_2: Linear<32, { Connect4::MAX_NUM_ACTIONS }>,
    v_1: Linear<60, 32>,
    v_2: Linear<32, 1>,
}

impl SlimC4Net {
    fn forward(&self, x: &[[[f32; 9]; 7]; 2]) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let x = self.c_1.forward::<9, 7, 4, 3>(x);
        let x = ReLU.apply_3d(&x);

        let x: [f32; 60] = unsafe { std::mem::transmute(x) };

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
    fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let state = env.state();
        let x = to_float(&state);
        let (logits, value) = self.forward(&x);
        let policy = Softmax.apply_1d(&logits);
        (policy, value)
    }
}
