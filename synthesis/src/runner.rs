use crate::config::{LearningConfig, ValueTarget};
use crate::data::ReplayBuffer;
use crate::game::Game;
use crate::mcts::MCTS;
use crate::policies::{Policy, PolicyWithCache, RolloutPolicy};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::StdRng;
use rand::SeedableRng;
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use std::collections::HashMap;

fn store_rewards<G: Game<N>, const N: usize>(
    cfg: &LearningConfig,
    buffer: &mut ReplayBuffer<G, N>,
    start_i: usize,
    mut r: f32,
) {
    // NOTE: buffer.vs[i] already has q value in it
    let num_turns = buffer.curr_steps() as f32 - start_i as f32;
    for (turn, i) in (start_i..buffer.curr_steps()).enumerate() {
        buffer.vs[i] = match cfg.value_target {
            ValueTarget::Q => buffer.vs[i],
            ValueTarget::Z => r,
            ValueTarget::ZplusQover2 => (buffer.vs[i] + r) / 2.0,
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

fn run_game<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<G, N>,
) {
    let mut game = G::new();
    let mut is_over = false;
    let mut search_policy = [0.0; N];
    let mut num_turns = cfg.num_random_actions;
    let start_i = buffer.vs.len();
    let start_player = game.player();

    while !is_over {
        let mut mcts = MCTS::with_capacity(
            cfg.num_explores + 1,
            cfg.c_puct,
            cfg.solve,
            cfg.fpu,
            policy,
            game.clone(),
        );

        if cfg.noisy_explore {
            mcts.add_dirichlet_noise(rng, cfg.alpha, cfg.noise_weight);
        }

        mcts.explore_n(cfg.num_explores);
        mcts.extract_search_policy(&mut search_policy);
        let best = mcts.best_action();
        buffer.add(&game.state(), &search_policy, mcts.extract_q());

        // assert!(
        //     search_policy[best.into()] > 0.0,
        //     "{:?} | {:?}",
        //     best,
        //     search_policy
        // );

        let action = if num_turns < cfg.num_random_actions {
            let n = rng.gen_range(0..game.iter_actions().count() as u8) as usize;
            game.iter_actions().nth(n).unwrap()
        } else if num_turns < cfg.sample_action_until && mcts.solution(&best).is_none() {
            let dist = WeightedIndex::new(&search_policy).unwrap();
            let choice = dist.sample(rng);
            G::Action::from(choice)
        } else {
            best
        };

        is_over = game.step(&action);
        num_turns += 1;
    }

    store_rewards(cfg, buffer, start_i, game.reward(start_player));
}

pub fn eval_against_random<G: Game<N>, P: Policy<G, N>, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    player: G::PlayerId,
) -> f32 {
    let mut game = G::new();
    let first_player = game.player();
    let mut opponent = StdRng::seed_from_u64(0);
    loop {
        let action = if game.player() == player {
            MCTS::exploit(
                cfg.num_explores,
                cfg.c_puct,
                cfg.solve,
                cfg.fpu,
                policy,
                game.clone(),
            )
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

pub fn eval_against_vanilla_mcts<G: Game<N>, P: Policy<G, N>, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    player: G::PlayerId,
    opponent_explores: usize,
    seed: u64,
) -> f32 {
    let mut game = G::new();
    let first_player = game.player();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rollout_policy = RolloutPolicy { rng: &mut rng };
    loop {
        let action = if game.player() == player {
            MCTS::exploit(
                cfg.num_explores,
                cfg.c_puct,
                cfg.solve,
                cfg.fpu,
                policy,
                game.clone(),
            )
        } else {
            MCTS::exploit(
                opponent_explores,
                cfg.c_puct,
                cfg.solve,
                cfg.fpu,
                &mut rollout_policy,
                game.clone(),
            )
        };

        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

pub fn mcts_vs_mcts<G: Game<N>, const N: usize>(
    cfg: &LearningConfig,
    player: G::PlayerId,
    p1_explores: usize,
    p2_explores: usize,
    seed: u64,
) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rollout_policy = RolloutPolicy { rng: &mut rng };
    let mut game = G::new();
    let first_player = game.player();
    loop {
        let action = MCTS::exploit(
            if game.player() == player {
                p1_explores
            } else {
                p2_explores
            },
            cfg.c_puct,
            cfg.solve,
            cfg.fpu,
            &mut rollout_policy,
            game.clone(),
        );
        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

pub fn gather_experience<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<G, N>,
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
