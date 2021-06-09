use super::traits::Policy;
use crate::env::Env;
use std::marker::PhantomData;

pub struct PolicyStorage<E: Env<N>, P: Policy<E, N>, const N: usize> {
    pub store: Vec<P>,
    pub names: Vec<String>,
    _env_marker: PhantomData<E>,
    _policy_marker: PhantomData<P>,
}

impl<E: Env<N>, P: Policy<E, N>, const N: usize> PolicyStorage<E, P, N> {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            store: Vec::with_capacity(n),
            names: Vec::with_capacity(n),
            _env_marker: PhantomData,
            _policy_marker: PhantomData,
        }
    }

    pub fn insert(&mut self, name: String, policy: P) {
        self.store.push(policy);
        self.names.push(name);
    }

    pub fn keep(&mut self, n: usize) {
        if self.names.len() >= n {
            let num_to_drop = self.names.len() - n;
            drop(self.names.drain(0..num_to_drop));
            drop(self.store.drain(0..num_to_drop));
        }
    }

    pub fn get(&mut self, name: &String) -> Option<&mut P> {
        match self.names.iter().position(|s| s == name) {
            Some(i) => Some(&mut self.store[i]),
            None => None,
        }
    }

    pub fn last(&self, n: usize) -> std::ops::Range<usize> {
        if self.names.len() < n {
            0..self.names.len()
        } else {
            (self.names.len() - n)..self.names.len()
        }
    }
}
