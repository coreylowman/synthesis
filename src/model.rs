use crate::data::tensor;
use crate::env::Env;
use crate::mcts::Policy;
use tch::{self, nn, Tensor};

pub struct ConvNet {
    device: tch::Device,
    conv_1: nn::Conv2D,
    // conv_2: nn::Conv2D,
    // conv_3: nn::Conv2D,
    fc_1: nn::Linear,
    // fc_2: nn::Linear,
    p: nn::Linear,
    v: nn::Linear,
}

impl ConvNet {
    pub fn new<E: Env>(vs: &nn::VarStore) -> Self {
        let cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let root = &vs.root();
        let dims = E::get_state_dims();
        assert!(dims.len() == 3);
        Self {
            device: vs.device(),
            conv_1: nn::conv2d(root / "conv_1", dims[0], 32, 3, cfg),
            // conv_2: nn::conv2d(root / "conv_2", 32, 32, 3, cfg),
            // conv_3: nn::conv2d(root / "conv_3", 32, 32, 3, cfg),
            fc_1: nn::linear(
                root / "fc_1",
                32 * dims[1] * dims[2],
                256,
                Default::default(),
            ),
            // fc_2: nn::linear(root / "fc_2", 256, 256, Default::default()),
            p: nn::linear(
                root / "p",
                256,
                E::MAX_NUM_ACTIONS as i64,
                Default::default(),
            ),
            v: nn::linear(root / "v", 256, 1, Default::default()),
        }
    }

    pub fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        // .apply(&self.conv_2)
        // .relu()
        // .apply(&self.conv_3)
        // .relu()
        // .apply(&self.fc_2);
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

impl<E: Env> Policy<E> for ConvNet {
    fn eval(&self, env: &E) -> (Vec<f32>, f32) {
        let xs = env.state();
        let t = tensor(&xs, &[1, 1, 6, 7], tch::Kind::Float);
        let (policy, value) = self.forward(&t);
        let policy = Vec::<f32>::from(&policy.squeeze1(0));
        let value = f32::from(&value);
        (policy, value)
    }
}

pub struct UniformRandomPolicy;
impl<E: Env> Policy<E> for UniformRandomPolicy {
    fn eval(&self, env: &E) -> (Vec<f32>, f32) {
        let xs = env.state();
        (
            vec![1.0 / (E::MAX_NUM_ACTIONS as f32); E::MAX_NUM_ACTIONS],
            0.0,
        )
    }
}
