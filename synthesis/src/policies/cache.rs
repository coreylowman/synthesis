use crate::game::Game;
use crate::policies::Policy;
use std::collections::HashMap;

pub struct PolicyWithCache<'a, G: Game<N>, P: Policy<G, N>, const N: usize> {
    pub policy: &'a mut P,
    pub cache: HashMap<G, ([f32; N], [f32; 3])>,
}

impl<'a, G: Game<N>, P: Policy<G, N>, const N: usize> PolicyWithCache<'a, G, P, N> {
    pub fn with_capacity(capacity: usize, policy: &'a mut P) -> Self {
        Self {
            policy,
            cache: HashMap::with_capacity(capacity),
        }
    }
}

impl<'a, G: Game<N>, P: Policy<G, N>, const N: usize> Policy<G, N>
    for PolicyWithCache<'a, G, P, N>
{
    fn eval(&mut self, game: &G) -> ([f32; N], [f32; 3]) {
        match self.cache.get(&game) {
            Some(pi_v) => *pi_v,
            None => {
                let pi_v = self.policy.eval(game);
                self.cache.insert(game.clone(), pi_v);
                pi_v
            }
        }
    }
}

pub struct OwnedPolicyWithCache<G: Game<N>, P: Policy<G, N>, const N: usize> {
    pub policy: P,
    pub cache: HashMap<G, ([f32; N], [f32; 3])>,
}

impl<G: Game<N>, P: Policy<G, N>, const N: usize> OwnedPolicyWithCache<G, P, N> {
    pub fn with_capacity(capacity: usize, policy: P) -> Self {
        Self {
            policy,
            cache: HashMap::with_capacity(capacity),
        }
    }
}

impl<G: Game<N>, P: Policy<G, N>, const N: usize> Policy<G, N> for OwnedPolicyWithCache<G, P, N> {
    fn eval(&mut self, game: &G) -> ([f32; N], [f32; 3]) {
        match self.cache.get(game) {
            Some(pi_v) => *pi_v,
            None => {
                let pi_v = self.policy.eval(game);
                self.cache.insert(game.clone(), pi_v);
                pi_v
            }
        }
    }
}
