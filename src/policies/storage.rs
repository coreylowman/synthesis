use super::traits::{NNPolicy, Policy};
use crate::envs::Env;
use std::collections::HashMap;
use tch::nn::VarStore;

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
