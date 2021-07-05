use crate::env::Env;
use crate::policies::Policy;
use std::collections::HashMap;

pub struct PolicyWithCache<'a, E: Env<N>, P: Policy<E, N>, const N: usize> {
    pub policy: &'a mut P,
    pub cache: HashMap<E::State, ([f32; N], f32)>,
}

impl<'a, E: Env<N>, P: Policy<E, N>, const N: usize> Policy<E, N> for PolicyWithCache<'a, E, P, N> {
    fn eval(&mut self, env: &E) -> ([f32; N], f32) {
        let state = env.state();
        match self.cache.get(&state) {
            Some(&(pi, v)) => (pi, v),
            None => {
                let (pi, v) = self.policy.eval(env);
                self.cache.insert(state, (pi, v));
                (pi, v)
            }
        }
    }
}

pub struct OwnedPolicyWithCache<E: Env<N>, P: Policy<E, N>, const N: usize> {
    pub policy: P,
    pub cache: HashMap<E::State, ([f32; N], f32)>,
}

impl<E: Env<N>, P: Policy<E, N>, const N: usize> Policy<E, N> for OwnedPolicyWithCache<E, P, N> {
    fn eval(&mut self, env: &E) -> ([f32; N], f32) {
        let state = env.state();
        match self.cache.get(&state) {
            Some(&(pi, v)) => (pi, v),
            None => {
                let (pi, v) = self.policy.eval(env);
                self.cache.insert(state, (pi, v));
                (pi, v)
            }
        }
    }
}
