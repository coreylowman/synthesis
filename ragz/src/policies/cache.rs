use crate::env::Env;
use crate::policies::Policy;
use std::collections::HashMap;

pub struct PolicyWithCache<'a, E: Env<N>, P: Policy<E, N>, const N: usize> {
    pub policy: &'a mut P,
    pub cache: HashMap<E::State, ([f32; N], f32)>,
}

impl<'a, E: Env<N>, P: Policy<E, N>, const N: usize> Policy<E, N> for PolicyWithCache<'a, E, P, N> {
    fn eval(&mut self, state: &E::State) -> ([f32; N], f32) {
        match self.cache.get(state) {
            Some(&(pi, v)) => (pi, v),
            None => {
                let (pi, v) = self.policy.eval(state);
                self.cache.insert(state.clone(), (pi, v));
                (pi, v)
            }
        }
    }
}
