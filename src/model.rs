use crate::env::Env;
use crate::mcts::Policy;
use tch::{self, nn, Tensor};

pub struct ConvNet {
    conv_1: nn::Conv2D,
    conv_2: nn::Conv2D,
    conv_3: nn::Conv2D,
    fc_1: nn::Linear,
    fc_2: nn::Linear,
    p: nn::Linear,
    v: nn::Linear,
}

impl ConvNet {
    pub fn new<E: Env>(vs: &nn::Path) -> Self {
        let cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        Self {
            conv_1: nn::conv2d(vs / "conv_1", 2, 32, 3, cfg),
            conv_2: nn::conv2d(vs / "conv_2", 32, 32, 3, cfg),
            conv_3: nn::conv2d(vs / "conv_3", 32, 32, 3, cfg),
            fc_1: nn::linear(vs / "fc_1", 32 * 6 * 7, 256, Default::default()),
            fc_2: nn::linear(vs / "fc_2", 256, 256, Default::default()),
            p: nn::linear(vs / "p", 256, E::MAX_NUM_ACTIONS as i64, Default::default()),
            v: nn::linear(vs / "v", 256, 1, Default::default()),
        }
    }

    pub fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs
            .apply(&self.conv_1)
            .relu()
            .apply(&self.conv_2)
            .relu()
            .apply(&self.conv_3)
            .relu()
            .flat_view()
            .apply(&self.fc_1)
            .relu()
            .apply(&self.fc_2);
        (
            xs.apply(&self.p).softmax(-1, tch::Kind::Float),
            xs.apply(&self.v).tanh(),
        )
    }
}

impl<E: Env> Policy<E> for ConvNet {
    fn eval(&self, env: &E) -> (Vec<f64>, f64) {
        let xs = env.state().unsqueeze(0);
        let (policy, value) = tch::no_grad(|| self.forward(&xs));
        let policy = Vec::<f64>::from(&policy.squeeze1(0));
        let value = value.double_value(&[]);
        (policy, value)
    }
}
