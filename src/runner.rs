use crate::data::ReplayBuffer;
use crate::envs::Env;
use crate::mcts::MCTS;
use crate::policies::Policy;
use crate::policy_impls::PolicyWithCache;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use rand_distr::Dirichlet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RolloutConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub temperature: f32,
    pub sample_action: bool,
    pub steps: usize,
    pub alpha: f32,
    pub noisy_explore: bool,
    pub c_puct: f32,
}

fn run_game<E: Env<N>, P: Policy<E, N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<E, N>,
) {
    let start_i = buffer.vs.len();
    let mut game = E::new();
    let start_player = game.player();
    let mut is_over = false;
    let mut search_policy = [0.0; N];
    let dirichlet = Dirichlet::new(&[cfg.alpha; N]).unwrap();

    while !is_over {
        let mut mcts =
            MCTS::<E, P, N>::with_capacity(cfg.num_explores, cfg.c_puct, policy, game.clone());

        if cfg.noisy_explore {
            mcts.add_noise(&dirichlet.sample(rng));
        }

        mcts.explore_n(cfg.num_explores);

        let root_node = mcts.root_node();

        // save timestep
        let mut total = 0.0;
        search_policy.fill(0.0);
        for &(action, child_id) in root_node.children.iter() {
            let child = mcts.get_node(child_id);
            let value = child.num_visits.powf(1.0 / cfg.temperature);
            search_policy[action.into()] = value;
            total += value;
        }
        for i in 0..N {
            search_policy[i] /= total;
        }

        buffer.add(mcts.root_state(), &search_policy, 0.0);

        let action = if cfg.sample_action {
            let dist = WeightedIndex::new(&search_policy).unwrap();
            let choice = dist.sample(rng);
            E::Action::from(choice)
        } else {
            mcts.best_action()
        };

        is_over = game.step(&action);
    }

    let mut r = game.reward(start_player);
    for i in start_i..buffer.vs.len() {
        buffer.vs[i] = r;
        r *= -1.0;
    }
    // if game.reward(game.player()) != 0.0 {
    //     assert!(game.reward(game.player()) == -1.0);
    //     assert!(buffer.vs[buffer.vs.len() - 1] == 1.0);
    // }
}

pub fn eval<E: Env<N>, P: Policy<E, N>, const N: usize>(
    cfg: &RolloutConfig,
    policy_a: &mut P,
    policy_b: &mut P,
) -> f32 {
    let mut game = E::new();
    let player = game.player();
    let mut mcts_a =
        MCTS::<E, P, N>::with_capacity(cfg.capacity, cfg.c_puct, policy_a, game.clone());
    let mut mcts_b =
        MCTS::<E, P, N>::with_capacity(cfg.capacity, cfg.c_puct, policy_b, game.clone());
    loop {
        let action = if game.player() == player {
            mcts_a.explore_n(cfg.num_explores);
            mcts_a.best_action()
        } else {
            mcts_b.explore_n(cfg.num_explores);
            mcts_b.best_action()
        };
        mcts_a.step_action(&action);
        mcts_b.step_action(&action);
        if game.step(&action) {
            break;
        }
    }
    game.print();
    game.reward(player)
}

pub fn gather_experience<E: Env<N>, P: Policy<E, N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<E, N>,
) {
    let mut cached_policy = PolicyWithCache {
        policy,
        cache: HashMap::with_capacity(cfg.steps * 2),
    };

    buffer.make_room(cfg.steps);
    let target = buffer.vs.len() + cfg.steps;
    let bar = ProgressBar::new(cfg.steps as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% {pos}/{len} {per_sec} {elapsed_precise}")
            .progress_chars("|| "),
    );
    let mut last = buffer.curr_steps();
    while buffer.vs.len() < target {
        buffer.new_game();
        run_game(cfg, &mut cached_policy, rng, buffer);
        bar.inc((buffer.curr_steps() - last) as u64);
        last = buffer.curr_steps();
    }
    bar.finish();
}
