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

    let batch_mean = 1.0 / (cfg.batch_size as f32);

    let vs = VarStore::new(tch::Device::Cpu);
    let mut policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, cfg.lr_schedule[0].1)?;
    opt.set_weight_decay(cfg.weight_decay);
    vs.save(models_dir.join(String::from("model_0.ot")))?;

    let mut dims = G::DIMS.to_owned();
    let mut buffer = ReplayBuffer::new(cfg.buffer_size);

    for i_iter in 0..cfg.num_iterations {
        // gather data
        {
            let _guard = tch::no_grad_guard();
            gather_experience(cfg, &mut policy, &mut rng, &mut buffer);
        }

        // convert to tensors
        let dedup = buffer.deduplicate();
        println!("Dedup {} -> {} steps", buffer.vs.len(), dedup.vs.len());
        dims[0] = dedup.vs.len() as i64;
        let states = tensor(&dedup.states, &dims, Kind::Float);
        let target_pis = tensor(&dedup.pis, &[dims[0], N as i64], Kind::Float);
        let target_vs = tensor(&dedup.vs, &[dims[0], 1], Kind::Float);

        let lr = cfg
            .lr_schedule
            .iter()
            .filter(|(i, _lr)| *i <= i_iter + 1)
            .last()
            .unwrap()
            .1;
        opt.set_lr(lr);
        println!("Using lr={}", lr);

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
    let mut cached_policy = PolicyWithCache::with_capacity(100 * cfg.games_per_train, policy);

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
    t: f32,
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
    let mut num_turns = 0;
    let start_player = game.player();
    let mut state_infos = Vec::with_capacity(100);

    while !is_over {
        let mut mcts = MCTS::with_capacity(
            cfg.num_explores + 1,
            cfg.learner_mcts_cfg,
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
        buffer.add(&game, &search_policy, 0.0);
        state_infos.push(StateInfo {
            turn: num_turns + 1,
            t: 0.0,
            q: mcts.extract_avg_value(),
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

    fill_state_info(&mut state_infos, game.reward(start_player));
    store_rewards(cfg, buffer, &state_infos);
}

fn fill_state_info(state_infos: &mut Vec<StateInfo>, mut reward: f32) {
    let num_turns = state_infos.len();
    for state_value in state_infos.iter_mut() {
        state_value.z = reward;
        state_value.t = state_value.turn as f32 / num_turns as f32;
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
            ValueTarget::QZaverage => 0.5 * (state.q + state.z),
            ValueTarget::QtoZ => state.z * state.t + state.q * (1.0 - state.t),
        };
    }
}
