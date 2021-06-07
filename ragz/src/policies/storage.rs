use super::traits::{NNPolicy, Policy};
use crate::env::Env;
use std::{collections::HashMap, marker::PhantomData};
use tch::nn::VarStore;

pub struct PolicyStorage<E: Env<N>, P: Policy<E, N> + NNPolicy<E, N>, const N: usize> {
    pub store: HashMap<String, VarStore>,
    pub names: Vec<String>,
    _env_marker: PhantomData<E>,
    _policy_marker: PhantomData<P>,
}

impl<E: Env<N>, P: Policy<E, N> + NNPolicy<E, N>, const N: usize> PolicyStorage<E, P, N> {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            store: HashMap::with_capacity(n),
            names: Vec::with_capacity(n),
            _env_marker: PhantomData,
            _policy_marker: PhantomData,
        }
    }

    pub fn insert(&mut self, name: &String, vs: &VarStore) {
        let mut stored_vs = VarStore::new(tch::Device::Cpu);
        let _policy = P::new(&stored_vs);
        stored_vs.copy(vs).unwrap();
        self.store.insert(name.clone(), stored_vs);
        self.names.push(name.clone());
    }

    pub fn get(&self, name: &String) -> P {
        let mut vs = VarStore::new(tch::Device::Cpu);
        let policy = P::new(&vs);
        let src_vs = self.store.get(name).unwrap();
        vs.copy(src_vs).unwrap();
        policy
    }

    pub fn last(&self, n: usize) -> &[String] {
        if self.names.len() < n {
            &self.names[..]
        } else {
            &self.names[(self.names.len() - n)..]
        }
    }
}
