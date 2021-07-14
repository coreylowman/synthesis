use crate::game::Game;
use crate::policies::Policy;
use std::collections::HashMap;

pub struct PolicyWithCache<'a, G: Game<N>, P: Policy<G, N>, const N: usize> {
    pub policy: &'a mut P,
    pub cache: HashMap<G::State, ([f32; N], f32)>,
}

impl<'a, G: Game<N>, P: Policy<G, N>, const N: usize> Policy<G, N>
    for PolicyWithCache<'a, G, P, N>
{
    fn eval(&mut self, game: &G) -> ([f32; N], f32) {
        let state = game.state();
        match self.cache.get(&state) {
            Some(pi_v) => *pi_v,
            None => {
                let pi_v = self.policy.eval(game);
                self.cache.insert(state, pi_v);
                pi_v
            }
        }
    }
}

pub struct OwnedPolicyWithCache<G: Game<N>, P: Policy<G, N>, const N: usize> {
    pub policy: P,
    pub cache: HashMap<G::State, ([f32; N], f32)>,
}

impl<G: Game<N>, P: Policy<G, N>, const N: usize> Policy<G, N> for OwnedPolicyWithCache<G, P, N> {
    fn eval(&mut self, game: &G) -> ([f32; N], f32) {
        let state = game.state();
        match self.cache.get(&state) {
            Some(pi_v) => *pi_v,
            None => {
                let pi_v = self.policy.eval(game);
                self.cache.insert(state, pi_v);
                pi_v
            }
        }
    }
}
