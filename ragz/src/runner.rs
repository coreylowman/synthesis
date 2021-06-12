use crate::data::ReplayBuffer;
use crate::env::Env;
use crate::mcts::MCTS;
use crate::policies::{Policy, PolicyWithCache};
use crate::vanilla_mcts::VanillaMCTS;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::StdRng;
use rand::SeedableRng;
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use rand_distr::Dirichlet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RolloutConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub sample_action_until: usize,
    pub steps: usize,
    pub alpha: f32,
    pub noisy_explore: bool,
    pub noise_weight: f32,
    pub c_puct: f32,
}

fn run_game<E: Env<N>, P: Policy<E, N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<E, N>,
) {
    let mut game = E::new();
    let mut is_over = false;
    let mut search_policy = [0.0; N];
    let mut num_turns = 0;
    let start_i = buffer.vs.len();
    let dirichlet = Dirichlet::new(&[cfg.alpha; N]).unwrap();
    let start_player = game.player();

    while !is_over {
        let mut mcts =
            MCTS::<E, P, N>::with_capacity(cfg.num_explores + 1, cfg.c_puct, policy, game.clone());

        if cfg.noisy_explore {
            mcts.add_noise(&dirichlet.sample(rng), cfg.noise_weight);
        }

        mcts.explore_n(cfg.num_explores);

        let root_node = mcts.root_node();

        // save timestep
        let mut total = 0.0;
        search_policy.fill(0.0);
        for &(action, child_id) in root_node.children.iter() {
            let child = mcts.get_node(child_id);
            search_policy[action.into()] = child.num_visits;
            total += child.num_visits;
        }
        for i in 0..N {
            search_policy[i] /= total;
        }

        // root_node.cum_value / root_node.num_visits,
        buffer.add(mcts.root_state(), &search_policy, 0.0);

        let action = if num_turns < cfg.sample_action_until {
            let dist = WeightedIndex::new(&search_policy).unwrap();
            let choice = dist.sample(rng);
            E::Action::from(choice)
        } else {
            mcts.best_action()
        };

        is_over = game.step(&action);
        num_turns += 1;
    }

    assert!(num_turns > cfg.sample_action_until);

    let mut r = game.reward(start_player);
    for i in start_i..buffer.vs.len() {
        buffer.vs[i] = r;
        r *= -1.0;
    }
}

pub fn eval_against_random<E: Env<N>, P: Policy<E, N>, const N: usize>(
    cfg: &RolloutConfig,
    policy: &mut P,
    player: E::PlayerId,
) -> f32 {
    let mut game = E::new();
    let first_player = game.player();
    let mut opponent = StdRng::seed_from_u64(0);
    loop {
        let action = if game.player() == player {
            let mut mcts = MCTS::<E, P, N>::with_capacity(
                cfg.num_explores + 1,
                cfg.c_puct,
                policy,
                game.clone(),
            );
            mcts.explore_n(cfg.num_explores);
            mcts.best_action()
        } else {
            let num_actions = game.iter_actions().count() as u8;
            let i = opponent.gen_range(0..num_actions) as usize;
            game.iter_actions().nth(i).unwrap()
        };

        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

pub fn eval_against_vanilla_mcts<E: Env<N>, P: Policy<E, N>, const N: usize>(
    cfg: &RolloutConfig,
    policy: &mut P,
    player: E::PlayerId,
    opponent_explores: usize,
) -> f32 {
    let mut game = E::new();
    let first_player = game.player();
    let mut rng = StdRng::seed_from_u64(0);
    loop {
        let action = if game.player() == player {
            let mut mcts = MCTS::<E, P, N>::with_capacity(
                cfg.num_explores + 1,
                cfg.c_puct,
                policy,
                game.clone(),
            );
            mcts.explore_n(cfg.num_explores);
            mcts.best_action()
        } else {
            let mut mcts =
                VanillaMCTS::<E, N>::with_capacity(opponent_explores + 1, game.clone(), &mut rng);
            mcts.explore_n(opponent_explores);
            mcts.best_action()
        };

        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
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
