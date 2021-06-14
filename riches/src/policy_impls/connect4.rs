use crate::envs::Connect4;
use ragz::prelude::*;
use slimnn::{self, Activation};
use tch::{self, nn, Tensor};

pub struct Connect4Net {
    p_1: nn::Linear,
    p_2: nn::Linear,
    p_3: nn::Linear,
    v_1: nn::Linear,
    v_2: nn::Linear,
    v_3: nn::Linear,
}

impl NNPolicy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn new(vs: &nn::VarStore) -> Self {
        let root = &vs.root();
        let state_dims = Connect4::get_state_dims();
        assert!(state_dims.len() == 4);
        assert!(state_dims[0] == 1);
        Self {
            p_1: nn::linear(
                root / "p_1",
                state_dims.iter().fold(1, |a, v| a * v),
                64,
                Default::default(),
            ),
            p_2: nn::linear(root / "p_2", 64, 32, Default::default()),
            p_3: nn::linear(
                root / "p_3",
                32,
                Connect4::MAX_NUM_ACTIONS as i64,
                Default::default(),
            ),
            v_1: nn::linear(
                root / "v_1",
                state_dims.iter().fold(1, |a, v| a * v),
                64,
                Default::default(),
            ),
            v_2: nn::linear(root / "v_2", 64, 32, Default::default()),
            v_3: nn::linear(root / "v_3", 32, 1, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs.flat_view();
        (
            xs.apply(&self.p_1)
                .relu()
                .apply(&self.p_2)
                .relu()
                .apply(&self.p_3),
            xs.apply(&self.v_1)
                .relu()
                .apply(&self.v_2)
                .relu()
                .apply(&self.v_3)
                .tanh(),
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
    p_1: slimnn::Linear<{ 2 * 7 * 9 }, 64>,
    p_2: slimnn::Linear<64, 32>,
    p_3: slimnn::Linear<32, { Connect4::MAX_NUM_ACTIONS }>,
    v_1: slimnn::Linear<{ 2 * 7 * 9 }, 64>,
    v_2: slimnn::Linear<64, 32>,
    v_3: slimnn::Linear<32, 1>,
}

impl SlimC4Net {
    fn forward(&self, x: &[f32; 2 * 7 * 9]) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let px = self.p_1.forward(&x);
        let px = slimnn::ReLU.apply_1d(&px);
        let px = self.p_2.forward(&px);
        let px = slimnn::ReLU.apply_1d(&px);
        let logits = self.p_3.forward(&px);

        let vx = self.v_1.forward(&x);
        let vx = slimnn::ReLU.apply_1d(&vx);
        let vx = self.v_2.forward(&vx);
        let vx = slimnn::ReLU.apply_1d(&vx);
        let value = self.v_3.forward(&vx)[0].tanh();

        (logits, value)
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for SlimC4Net {
    fn eval(
        &mut self,
        xs: &<Connect4 as Env<{ Connect4::MAX_NUM_ACTIONS }>>::State,
    ) -> ([f32; Connect4::MAX_NUM_ACTIONS], f32) {
        let bool_x: [bool; 2 * 7 * 9] = unsafe { std::mem::transmute(*xs) };
        let mut x = [0.0; 2 * 7 * 9];
        for i in 0..(2 * 7 * 9) {
            x[i] = if bool_x[i] { 1.0 } else { 0.0 };
        }
        let (logits, value) = self.forward(&x);
        let policy = slimnn::Softmax.apply_1d(&logits);
        (policy, value)
    }
}
