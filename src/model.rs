use std::marker::PhantomData;

use crate::data::tensor;
use crate::env::Env;
use crate::mcts::Policy;
use tch::{self, nn, Tensor};

pub trait NNPolicy<E: Env> {
    fn new(vs: &nn::VarStore) -> Self;
    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor);
}

pub struct ConvNet<E: Env> {
    state_dims: Vec<i64>,
    conv_1: nn::Conv2D,
    fc_1: nn::Linear,
    p: nn::Linear,
    v: nn::Linear,
    _marker: PhantomData<E>,
}

impl<E: Env> NNPolicy<E> for ConvNet<E> {
    fn new(vs: &nn::VarStore) -> Self {
        let cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let root = &vs.root();
        let mut state_dims = E::get_state_dims();
        assert!(state_dims.len() == 3);
        state_dims.insert(0, 1);
        Self {
            conv_1: nn::conv2d(root / "conv_1", state_dims[1], 32, 3, cfg),
            fc_1: nn::linear(
                root / "fc_1",
                32 * state_dims[2] * state_dims[3],
                64,
                Default::default(),
            ),
            p: nn::linear(
                root / "p",
                64,
                E::MAX_NUM_ACTIONS as i64,
                Default::default(),
            ),
            v: nn::linear(root / "v", 64, 1, Default::default()),
            state_dims,
            _marker: PhantomData,
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs
            .apply(&self.conv_1)
            .relu()
            .flat_view()
            .apply(&self.fc_1)
            .relu();
        (
            xs.apply(&self.p).softmax(-1, tch::Kind::Float),
            xs.apply(&self.v).tanh(),
        )
    }
}

impl<E: Env> Policy<E> for ConvNet<E> {
    fn eval(&mut self, xs: &Vec<f32>) -> (Vec<f32>, f32) {
        let t = tensor(&xs, &self.state_dims, tch::Kind::Float);
        let (policy, value) = self.forward(&t);
        let policy = Vec::<f32>::from(&policy);
        // assert!(policy.len() == E::MAX_NUM_ACTIONS);
        let value = f32::from(&value);
        (policy, value)
    }
}
pub struct UniformRandomPolicy;
impl<E: Env> Policy<E> for UniformRandomPolicy {
    fn eval(&mut self, xs: &Vec<f32>) -> (Vec<f32>, f32) {
        (
            vec![1.0 / (E::MAX_NUM_ACTIONS as f32); E::MAX_NUM_ACTIONS],
            0.0,
        )
    }
}
