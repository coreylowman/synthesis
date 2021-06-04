use crate::data::tensor;
use crate::envs::{Connect4, Env};
use crate::policies::{NNPolicy, Policy};
use tch::{self, nn, Tensor};

pub struct Connect4Net {
    fc_1: nn::Linear,
    fc_2: nn::Linear,
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
            fc_1: nn::linear(
                root / "fc_1",
                state_dims.iter().fold(1, |a, v| a * v),
                32,
                Default::default(),
            ),
            fc_2: nn::linear(root / "fc_2", 32, 32, Default::default()),
            p_1: nn::linear(root / "p_1", 32, 32, Default::default()),
            p_2: nn::linear(
                root / "p_2",
                32,
                Connect4::MAX_NUM_ACTIONS as i64,
                Default::default(),
            ),
            v_1: nn::linear(root / "v_1", 32, 32, Default::default()),
            v_2: nn::linear(root / "v_2", 32, 1, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs
            .flat_view()
            .apply(&self.fc_1)
            .relu()
            .apply(&self.fc_2)
            .relu();
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
