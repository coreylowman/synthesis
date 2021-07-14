use crate::config::{LearningConfig, ValueTarget};
use crate::data::*;
use crate::game::Game;
use crate::mcts::MCTS;
use crate::policies::{NNPolicy, Policy, PolicyWithCache};
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};
use std::collections::HashMap;
use std::default::Default;
use tch::{
    kind::Kind,
    nn::{Adam, OptimizerConfig, VarStore},
};

pub fn learner<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: &LearningConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&cfg.logs)?;
    let models_dir = cfg.logs.join("models");

    std::fs::create_dir(&models_dir)?;
    save(&cfg.logs, "cfg.json", cfg)?;
    save_str(&cfg.logs, "env_name", &G::NAME.into())?;
    save_str(&cfg.logs, "git_hash", &git_hash()?)?;
    save_str(&cfg.logs, "git_diff.patch", &git_diff()?)?;

    tch::manual_seed(cfg.seed as i64);
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let vs = VarStore::new(tch::Device::Cpu);
    let mut policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, cfg.lr)?;
    opt.set_weight_decay(cfg.weight_decay);

    let mut dims = G::get_state_dims();

    vs.save(models_dir.join(String::from("model_0.ot")))?;

    let mut buffer = ReplayBuffer::new(cfg.buffer_size);

    for i_iter in 0..cfg.num_iterations {
        // gather data
        {
            let _guard = tch::no_grad_guard();
            gather_experience(cfg, &mut policy, &mut rng, &mut buffer);
        }

        // convert to tensors
        let flat_batch = buffer.deduplicate();
        println!(
            "Deduplicated {} -> {} steps",
            buffer.vs.len(),
            flat_batch.vs.len()
        );
        dims[0] = buffer.vs.len() as i64;
        let states = tensor(&buffer.states, &dims, Kind::Bool).to_kind(tch::Kind::Float);
        let target_pis = tensor(&buffer.pis, &[dims[0], N as i64], Kind::Float);
        let target_vs = tensor(&buffer.vs, &[dims[0], 1], Kind::Float);

        let batch_mean = 1.0 / (cfg.batch_size as f32);

        // train
        for _i_epoch in 0..cfg.num_epochs {
            let sampler =
                BatchRandSampler::new(&states, &target_pis, &target_vs, cfg.batch_size, true);

            let mut epoch_loss = [0.0, 0.0];
            for (state, target_pi, target_v) in sampler {
                assert_eq!(state.size()[0], cfg.batch_size);

                let (logits, v) = policy.forward(&state);
                assert_eq!(logits.size(), target_pi.size());
                assert_eq!(v.size(), target_v.size());

                let log_pi = logits.log_softmax(-1, Kind::Float);
                let pi_loss = batch_mean * log_pi.kl_div(&target_pi, tch::Reduction::Sum, false);
                let v_loss = v.mse_loss(&target_v, tch::Reduction::Mean);

                let loss = &pi_loss + &v_loss;
                opt.backward_step(&loss);

                epoch_loss[0] += f32::from(&pi_loss);
                epoch_loss[1] += f32::from(&v_loss);
            }
            epoch_loss[0] *= (cfg.batch_size as f32) / (dims[0] as f32);
            epoch_loss[1] *= (cfg.batch_size as f32) / (dims[0] as f32);
            println!("{} {:?}", _i_epoch, epoch_loss);
        }

        // save latest weights
        vs.save(models_dir.join(format!("model_{}.ot", i_iter + 1)))?;

        println!(
            "Finished iteration {} | {} games played / {} steps taken | {} games / {} steps in replay buffer",
            i_iter + 1,
            buffer.total_games_played(),
            buffer.total_steps(),
            buffer.curr_games(),
            buffer.curr_steps(),
        );
    }

    Ok(())
}
fn gather_experience<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
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
