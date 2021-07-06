use crate::env::Env;
use crate::policies::Policy;
use rand::Rng;

pub struct RolloutPolicy<'a, R: Rng> {
    pub rng: &'a mut R,
}
impl<'a, E: Env<N>, R: Rng, const N: usize> Policy<E, N> for RolloutPolicy<'a, R> {
    fn eval(&mut self, env: &E) -> ([f32; N], f32) {
        let player = env.player();
        let mut rollout_env = env.clone();
        let mut is_over = env.is_over();
        while !is_over {
            let actions = rollout_env.iter_actions();
            let num_actions = actions.count() as u8;
            let i = self.rng.gen_range(0..num_actions);
            let action = rollout_env.iter_actions().nth(i as usize).unwrap();
            is_over = rollout_env.step(&action);
        }
        ([0.0; N], rollout_env.reward(player))
    }
}
