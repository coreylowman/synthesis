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
        let outcome_logits = ts.pop().unwrap();
        let policy_logits = ts.pop().unwrap();
        (policy_logits, outcome_logits)
    }
}

impl Policy<Connect4, { Connect4::MAX_NUM_ACTIONS }> for Connect4Net {
    fn eval(&mut self, env: &Connect4) -> ([f32; Connect4::MAX_NUM_ACTIONS], [f32; 3]) {
        let xs = env.features();
        let t = tensor(&xs, Connect4::DIMS, tch::Kind::Float);
        let (logits, value) = self.forward(&t);
        let mut policy = [0.0f32; Connect4::MAX_NUM_ACTIONS];
        logits.copy_data(&mut policy, Connect4::MAX_NUM_ACTIONS);
        let mut outcomes = [0.0f32; 3];
        value
            .softmax(-1, tch::Kind::Float)
            .copy_data(&mut outcomes, 3);
        (policy, outcomes)
    }
}
