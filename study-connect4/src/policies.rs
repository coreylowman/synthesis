use crate::connect4::Connect4;
use synthesis::prelude::*;
use tch::{self, nn, Tensor};

pub struct Connect4Net {
    l_1: nn::Linear,
    l_2: nn::Linear,
    l_3: nn::Linear,
    l_4: nn::Linear,
    l_5: nn::Linear,
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
            l_4: nn::linear(root / "l_4", 64, 48, Default::default()),
            l_5: nn::linear(root / "l_5", 48, 12, Default::default()),
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
            .apply(&self.l_4)
            .relu()
            .apply(&self.l_5);
        let mut ts = xs.split_with_sizes(&[9, 3], -1);
        let value_ps = ts.pop().unwrap().softmax(-1, tch::Kind::Float);
        let t = Tensor::of_slice(&[-1.0, 0.0, 1.0]);
        let value = (value_ps * t).sum1(&[-1], true, tch::Kind::Float);
        // let mut ts = xs.split_with_sizes(&[9, 1], -1);
        // let value = ts.pop().unwrap();
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
