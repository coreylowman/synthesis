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

struct StateInfo {
    turn: usize,
    q: f32,
    z: f32,
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
    let mut num_turns = cfg.num_random_actions; // TODO fix this, should be 0.0
    let start_player = game.player();
    let mut state_infos = Vec::with_capacity(100);

    while !is_over {
        let mut mcts = MCTS::with_capacity(
            cfg.num_explores + 1,
            cfg.c_puct,
            cfg.solve,
            cfg.fpu,
            policy,
            game.clone(),
        );

        // explore
        if cfg.noisy_explore {
            mcts.add_dirichlet_noise(rng, cfg.alpha, cfg.noise_weight);
        }
        mcts.explore_n(cfg.num_explores);

        // store in buffer
        mcts.extract_search_policy(&mut search_policy);
        buffer.add(&game.state(), &search_policy, 0.0);
        state_infos.push(StateInfo {
            turn: num_turns + 1,
            q: mcts.extract_q(),
            z: 0.0,
        });

        // pick action
        // TODO flip a bool if mcts.solution is some instead of checking again
        let best = mcts.best_action();
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

    fill_zs(&mut state_infos, game.reward(start_player));
    store_rewards(cfg, buffer, &state_infos);
}

fn fill_zs(state_infos: &mut Vec<StateInfo>, mut reward: f32) {
    for state_value in state_infos.iter_mut() {
        state_value.z = reward;
        reward = -reward;
    }
}

fn store_rewards<G: Game<N>, const N: usize>(
    cfg: &LearningConfig,
    buffer: &mut ReplayBuffer<G, N>,
    state_infos: &Vec<StateInfo>,
) {
    let num_turns = state_infos.len();
    let start_i = buffer.curr_steps() - num_turns;
    let end_i = buffer.curr_steps();
    for (buffer_value, state) in buffer.vs[start_i..end_i].iter_mut().zip(state_infos) {
        *buffer_value = match cfg.value_target {
            ValueTarget::Q => state.q,
            ValueTarget::Z => state.z,
            ValueTarget::ZplusQover2 => (state.z + state.q) / 2.0,
            ValueTarget::Interpolate => {
                let t = state.turn as f32 / num_turns as f32;
                state.z * t + state.q * (1.0 - t)
            }
            ValueTarget::QForSamples => {
                if state.turn <= cfg.sample_action_until {
                    state.q
                } else {
                    state.z
                }
            }
            ValueTarget::InterpolateForSamples => {
                let t = state.turn as f32 / num_turns as f32;
                if state.turn <= cfg.sample_action_until {
                    state.z * t + state.q * (1.0 - t)
                } else {
                    state.z
                }
            }
            ValueTarget::SteepInterpolateForSamples => {
                let t = state.turn as f32 / cfg.sample_action_until as f32;
                if state.turn <= cfg.sample_action_until {
                    state.z * t + state.q * (1.0 - t)
                } else {
                    state.z
                }
            }
        };
    }
}
