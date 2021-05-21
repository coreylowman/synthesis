use crate::env::Env;
use crate::mcts::Policy;
use tch::{self, nn};

pub struct NNPolicy<M: nn::Module> {
    pub model: M,
}

impl<E: Env, M: nn::Module> Policy<E> for NNPolicy<M> {
    fn eval(&self, env: &E) -> (Vec<f64>, f64) {
        let out = tch::no_grad(|| self.model.forward(&env.state()));
        let dim = *out.size().last().unwrap();
        let splits = out.split_with_sizes(&[dim - 1, 1], -1);
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[1].size(), vec![1]);
        let policy = Vec::<f64>::from(&splits[0]);
        let value = splits[1].double_value(&[]);
        (policy, value)
    }
}
