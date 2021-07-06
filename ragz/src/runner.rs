use crate::data::ReplayBuffer;
use crate::env::Env;
use crate::mcts::MCTS;
use crate::policies::{Policy, PolicyWithCache, RolloutPolicy};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::StdRng;
use rand::SeedableRng;
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use rand_distr::Dirichlet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ValueTarget {
    Z,                          // Outcome of game {-1, 0, 1}
    Q,                          // Avg Value found while searching
    QZAverage,                  // (Q + Z) / 2
    Interpolate,                // interpolate between Z and Q
    QForSamples,                // Q if action is sampled, Z if action is exploit
    InterpolateForSamples,      // Interp if action is sampled, Z if action is exploit
    SteepInterpolateForSamples, // Interp to end of sampling if action is sampled, Z if action is exploit
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RolloutConfig {
    pub buffer_size: usize,
    pub games_to_keep: usize,
    pub games_per_train: usize,
    pub num_explores: usize,
    pub sample_action_until: usize,
    pub alpha: f32,
    pub noisy_explore: bool,
    pub noise_weight: f32,
    pub c_puct: f32,
    pub solve: bool,
    pub value_target: ValueTarget,
}

fn store_rewards<E: Env<N>, const N: usize>(
    cfg: &RolloutConfig,
    buffer: &mut ReplayBuffer<E, N>,
    start_i: usize,
    mut r: f32,
) {
    // NOTE: buffer.vs[i] already has q value in it
    let num_turns = buffer.curr_steps() as f32 - start_i as f32;
    for (turn, i) in (start_i..buffer.curr_steps()).enumerate() {
        buffer.vs[i] = match cfg.value_target {
            ValueTarget::Q => buffer.vs[i],
            ValueTarget::Z => r,
            ValueTarget::QZAverage => (buffer.vs[i] + r) / 2.0,
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
            ValueTarget::SteepInterpolateForSamples => {
                let t = (turn + 1) as f32 / cfg.sample_action_until as f32;
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
        let mut mcts = MCTS::<E, P, N>::with_capacity(
            cfg.num_explores + 1,
            cfg.c_puct,
            cfg.solve,
            policy,
            game.clone(),
        );

        if cfg.noisy_explore {
            mcts.add_noise(&dirichlet.sample(rng), cfg.noise_weight);
        }

        mcts.explore_n(cfg.num_explores);
        mcts.extract_search_policy(&mut search_policy);
        let best = mcts.best_action();
        buffer.add(&game.state(), &search_policy, mcts.extract_q());

        let action = if num_turns < cfg.sample_action_until && mcts.outcome(&best).is_none() {
            let dist = WeightedIndex::new(&search_policy).unwrap();
            let choice = dist.sample(rng);
            E::Action::from(choice)
        } else {
            best
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
        let mut rollout_policy = RolloutPolicy { rng };
        let mut mcts = MCTS::<E, RolloutPolicy<R>, N>::with_capacity(
            cfg.num_explores + 1,
            cfg.c_puct,
            cfg.solve,
            &mut rollout_policy,
            game.clone(),
        );

        mcts.explore_n(cfg.num_explores);
        mcts.extract_search_policy(&mut search_policy);
        let best = mcts.best_action();
        buffer.add(&game.state(), &search_policy, mcts.extract_q());

        let action = if num_turns < cfg.sample_action_until && mcts.outcome(&best).is_none() {
            let dist = WeightedIndex::new(&search_policy).unwrap();
            let choice = dist.sample(rng);
            E::Action::from(choice)
        } else {
            best
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
                cfg.solve,
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
    let mut rollout_policy = RolloutPolicy { rng: &mut rng };
    loop {
        let action = if game.player() == player {
            let mut mcts = MCTS::<E, P, N>::with_capacity(
                cfg.num_explores + 1,
                cfg.c_puct,
                cfg.solve,
                policy,
                game.clone(),
            );
            mcts.explore_n(cfg.num_explores);
            mcts.best_action()
        } else {
            let mut mcts = MCTS::<E, RolloutPolicy<StdRng>, N>::with_capacity(
                opponent_explores + 1,
                cfg.c_puct,
                cfg.solve,
                &mut rollout_policy,
                game.clone(),
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

pub fn mcts_vs_mcts<E: Env<N>, const N: usize>(
    cfg: &RolloutConfig,
    player: E::PlayerId,
    p1_explores: usize,
    p2_explores: usize,
    seed: u64,
) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rollout_policy = RolloutPolicy { rng: &mut rng };
    let mut game = E::new();
    let first_player = game.player();
    loop {
        let explores = if game.player() == player {
            p1_explores
        } else {
            p2_explores
        };
        let mut mcts = MCTS::<E, RolloutPolicy<StdRng>, N>::with_capacity(
            explores + 1,
            cfg.c_puct,
            cfg.solve,
            &mut rollout_policy,
            game.clone(),
        );
        mcts.explore_n(explores);
        let action = mcts.best_action();
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
        cache: HashMap::with_capacity(100 * cfg.games_per_train),
    };

    buffer.keep_last_n_games(cfg.games_to_keep - cfg.games_per_train);
    let bar = ProgressBar::new(cfg.games_per_train as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% {pos}/{len} {per_sec} {elapsed_precise}")
            .progress_chars("|| "),
    );
    for _ in 0..cfg.games_per_train {
        buffer.new_game();
        run_game(cfg, &mut cached_policy, rng, buffer);
        bar.inc(1);
    }
    bar.finish();
}

pub fn fill_buffer<E: Env<N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    rng: &mut R,
    buffer: &mut ReplayBuffer<E, N>,
) {
    let target = cfg.games_to_keep - cfg.games_per_train;
    let bar = ProgressBar::new(target as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% {pos}/{len} {per_sec} {elapsed_precise}")
            .progress_chars("|| "),
    );
    for _ in 0..target {
        buffer.new_game();
        run_vanilla_mcts_game(cfg, rng, buffer);
        bar.inc(1);
    }
    bar.finish();
}
