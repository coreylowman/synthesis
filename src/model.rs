use crate::data::tensor;
use crate::envs::Env;
use crate::mcts::Policy;
use std::collections::HashMap;
use std::marker::PhantomData;
use tch::{
    self,
    nn::{self, VarStore},
    Tensor,
};

pub trait NNPolicy<E: Env> {
    fn new(vs: &nn::VarStore) -> Self;
    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor);
}

pub struct PolicyStorage {
    pub store: HashMap<String, VarStore>,
    pub names: Vec<String>,
}

impl PolicyStorage {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            store: HashMap::with_capacity(n),
            names: Vec::with_capacity(n),
        }
    }

    pub fn insert(&mut self, name: &String, vs: &VarStore) {
        let mut stored_vs = VarStore::new(tch::Device::Cpu);
        stored_vs.copy(vs).unwrap();
        self.store.insert(name.clone(), stored_vs);
        self.names.push(name.clone());
    }

    pub fn get<E: Env, P: Policy<E> + NNPolicy<E>>(&self, name: &String) -> P {
        P::new(self.store.get(name).unwrap())
    }

    pub fn last(&self, n: usize) -> &[String] {
        if self.names.len() < n {
            &self.names[..]
        } else {
            &self.names[(self.names.len() - n)..]
        }
    }
}

pub struct ConvNet<E: Env> {
    conv_1: nn::Conv2D,
    fc_1: nn::Linear,
    fc_2: nn::Linear,
    p_1: nn::Linear,
    p_2: nn::Linear,
    v_1: nn::Linear,
    v_2: nn::Linear,
    _marker: PhantomData<E>,
}

impl<E: Env> NNPolicy<E> for ConvNet<E> {
    fn new(vs: &nn::VarStore) -> Self {
        let root = &vs.root();
        let state_dims = E::get_state_dims();
        assert!(state_dims.len() == 4);
        assert!(state_dims[0] == 1);
        Self {
            conv_1: nn::conv2d(root / "conv_1", state_dims[1], 256, 4, Default::default()),
            fc_1: nn::linear(
                root / "fc_1",
                256 * (state_dims[2] - 3) * (state_dims[3] - 3),
                64,
                Default::default(),
            ),
            fc_2: nn::linear(root / "fc_2", 64, 64, Default::default()),
            p_1: nn::linear(root / "p_1", 64, 64, Default::default()),
            p_2: nn::linear(
                root / "p_2",
                64,
                E::MAX_NUM_ACTIONS as i64,
                Default::default(),
            ),
            v_1: nn::linear(root / "v_1", 64, 64, Default::default()),
            v_2: nn::linear(root / "v_2", 64, 1, Default::default()),
            _marker: PhantomData,
        }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs
            .apply(&self.conv_1)
            .relu()
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

impl<E: Env> Policy<E> for ConvNet<E> {
    fn eval(&mut self, xs: &Vec<f32>) -> (Vec<f32>, f32) {
        let t = tensor(&xs, &E::get_state_dims(), tch::Kind::Float);
        let (logits, value) = self.forward(&t);
        let policy = logits.softmax(-1, tch::Kind::Float);
        let policy = Vec::<f32>::from(&policy);
        // assert!(policy.len() == E::MAX_NUM_ACTIONS);
        let value = f32::from(&value);
        (policy, value)
    }
}
