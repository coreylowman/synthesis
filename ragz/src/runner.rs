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
pub enum ValueTarget {
    Z,                     // Outcome of game {-1, 0, 1}
    Q,                     // Avg Value found while searching
    Interpolate,           // interpolate between Z and Q
    QForSamples,           // Q if action is sampled, Z if action is exploit
    InterpolateForSamples, // Interp if action is sampled, Z if action is exploit
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RolloutConfig {
    pub buffer_size: usize,
    pub num_explores: usize,
    pub sample_action_until: usize,
    pub steps: usize,
    pub alpha: f32,
    pub noisy_explore: bool,
    pub noise_weight: f32,
    pub c_puct: f32,
    pub value_target: ValueTarget,
}

fn store_rewards<E: Env<N>, const N: usize>(
    cfg: &RolloutConfig,
    buffer: &mut ReplayBuffer<E, N>,
    start_i: usize,
    mut r: f32,
) {
    // NOTE: buffer.vs[i] already has q value in it
    let num_turns = buffer.vs.len() as f32 - start_i as f32;
    for (turn, i) in (start_i..buffer.vs.len()).enumerate() {
        buffer.vs[i] = match cfg.value_target {
            ValueTarget::Q => buffer.vs[i],
            ValueTarget::Z => r,
            ValueTarget::Interpolate => {
                let t = (turn + 1) as f32 / num_turns;
                r * t + buffer.vs[i] * (1.0 - t)
            }
            ValueTarget::QForSamples => {
                if turn < cfg.sample_action_until {
                    buffer.vs[i]
                } else {
                    r
                }
            }
            ValueTarget::InterpolateForSamples => {
                let t = (turn + 1) as f32 / num_turns;
                if turn < cfg.sample_action_until {
                    r * t + buffer.vs[i] * (1.0 - t)
                } else {
                    r
                }
            }
        };

        r = -r;
    }
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
        search_policy.fill(0.0);
        for &(action, child_id) in root_node.children.iter() {
            let child = mcts.get_node(child_id);
            search_policy[action.into()] = child.num_visits / cfg.num_explores as f32;
        }

        // root_node.cum_value / root_node.num_visits,
        buffer.add(
            mcts.root_state(),
            &search_policy,
            root_node.cum_value / cfg.num_explores as f32,
        );

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

    store_rewards(cfg, buffer, start_i, game.reward(start_player));
}

fn run_vanilla_mcts_game<E: Env<N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    rng: &mut R,
    buffer: &mut ReplayBuffer<E, N>,
) {
    let mut game = E::new();
    let mut is_over = false;
    let mut num_turns = 0;
    let mut search_policy = [0.0; N];
    let start_i = buffer.vs.len();
    let start_player = game.player();

    while !is_over {
        let mut mcts =
            VanillaMCTS::<E, R, N>::with_capacity(cfg.num_explores + 1, game.clone(), rng);

        mcts.explore_n(cfg.num_explores);

        let root_node = mcts.root_node();
        search_policy.fill(0.0);
        for &(action, child_id) in root_node.children.iter() {
            let child = mcts.get_node(child_id);
            search_policy[action.into()] = child.num_visits / cfg.num_explores as f32;
        }

        buffer.add(
            &game.state(),
            &search_policy,
            root_node.cum_value / cfg.num_explores as f32,
        );

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

    store_rewards(cfg, buffer, start_i, game.reward(start_player));
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
    seed: u64,
) -> f32 {
    let mut game = E::new();
    let first_player = game.player();
    let mut rng = StdRng::seed_from_u64(seed);
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
            let mut mcts = VanillaMCTS::<E, StdRng, N>::with_capacity(
                opponent_explores + 1,
                game.clone(),
                &mut rng,
            );
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

pub fn fill_buffer<E: Env<N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    rng: &mut R,
    buffer: &mut ReplayBuffer<E, N>,
) {
    let target = cfg.buffer_size - cfg.steps;
    let bar = ProgressBar::new(target as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% {pos}/{len} {per_sec} {elapsed_precise}")
            .progress_chars("|| "),
    );
    let mut last = buffer.curr_steps();
    while buffer.vs.len() < target {
        run_vanilla_mcts_game(cfg, rng, buffer);
        bar.inc((buffer.curr_steps() - last) as u64);
        last = buffer.curr_steps();
    }
    bar.finish();
}
