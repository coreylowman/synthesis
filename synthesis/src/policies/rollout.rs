use crate::game::Game;
use crate::policies::Policy;
use rand::Rng;

pub struct RolloutPolicy<'a, R: Rng> {
    pub rng: &'a mut R,
}
impl<'a, G: Game<N>, R: Rng, const N: usize> Policy<G, N> for RolloutPolicy<'a, R> {
    fn eval(&mut self, game: &G) -> ([f32; N], f32) {
        let player = game.player();
        let mut rollout_game = game.clone();
        let mut is_over = game.is_over();
        while !is_over {
            let actions = rollout_game.iter_actions();
            let num_actions = actions.count() as u8;
            let i = self.rng.gen_range(0..num_actions);
            let action = rollout_game.iter_actions().nth(i as usize).unwrap();
            is_over = rollout_game.step(&action);
        }
        ([0.0; N], rollout_game.reward(player))
    }
}
