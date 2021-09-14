use crate::config::{LearningConfig, RolloutNoise, ValueTarget};
use crate::data::*;
use crate::game::{Game, Outcome};
use crate::mcts::MCTS;
use crate::policies::{NNPolicy, Policy, PolicyWithCache};
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand::{distributions::Distribution, distributions::WeightedIndex};
use std::default::Default;
use tch::{
    kind::Kind,
    nn::{Adam, OptimizerConfig, VarStore},
};

pub fn alpha_zero<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: &LearningConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // set up directory structure
    std::fs::create_dir_all(&cfg.logs)?;
    let models_dir = cfg.logs.join("models");
    std::fs::create_dir(&models_dir)?;
    save(&cfg.logs, "cfg.json", cfg)?;
    save_str(&cfg.logs, "env_name", &G::NAME.into())?;
    save_str(&cfg.logs, "git_hash", &git_hash()?)?;
    save_str(&cfg.logs, "git_diff.patch", &git_diff()?)?;

    // seed rngs
    tch::manual_seed(cfg.seed as i64);
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    // init policy
    let vs = VarStore::new(tch::Device::Cpu);
    let mut policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, cfg.lr_schedule[0].1)?;
    if cfg.weight_decay > 0.0 {
        opt.set_weight_decay(cfg.weight_decay);
    }
    vs.save(models_dir.join(String::from("model_0.ot")))?;

    // init replay buffer
    let mut buffer = ReplayBuffer::new(cfg.buffer_size);

    // start learning!
    let mut dims = G::DIMS.to_owned();
    let batch_mean = 1.0 / (cfg.batch_size as f32);
    for i_iter in 0..cfg.num_iterations {
        // gather data
        {
            let _guard = tch::no_grad_guard();
            gather_experience(cfg, &mut policy, &mut rng, &mut buffer);
        }

        // convert buffer data to tensors
        let dedup = buffer.deduplicate();
        println!("Dedup {} -> {} steps", buffer.vs.len(), dedup.vs.len());
        dims[0] = dedup.vs.len() as i64;
        let states = tensor(&dedup.states, &dims, Kind::Float);
        let target_pis = tensor(&dedup.pis, &[dims[0], N as i64], Kind::Float);
        let target_vs = tensor(&dedup.vs, &[dims[0], 1], Kind::Float);

        // calculate lr from schedule
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
                // assert_eq!(state.size()[0], cfg.batch_size);

                let (logits, v) = policy.forward(&state);
                // assert_eq!(logits.size(), target_pi.size());
                // assert_eq!(v.size(), target_v.size());

                let log_pi = logits.log_softmax(-1, Kind::Float);
                let pi_loss = batch_mean * log_pi.kl_div(&target_pi, tch::Reduction::Sum, false);
                let v_loss = v.mse_loss(&target_v, tch::Reduction::Mean);

                let loss = cfg.policy_weight * &pi_loss + cfg.value_weight * &v_loss;
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
        states.write_npy(cfg.logs.join("latest_states.npy"))?;
        target_pis.write_npy(cfg.logs.join("latest_pis.npy"))?;
        target_vs.write_npy(cfg.logs.join("latest_vs.npy"))?;

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
    let num_workers = cfg.num_threads + 1;
    let num_games_per_thread = cfg.games_per_train / num_workers;
    let num_games_in_main_thread = num_games_per_thread + cfg.games_per_train % num_workers;
    assert_eq!(
        num_games_per_thread * cfg.num_threads + num_games_in_main_thread,
        cfg.games_per_train
    );

    for _ in 0..cfg.num_threads {
        todo!("num_threads > 0 unsupported");
        // TODO initialize multi progress bar
        // TODO spawn a thread
    }

    buffer.keep_last_n_games(cfg.games_to_keep - cfg.games_per_train);
    let bar = ProgressBar::new(cfg.games_per_train as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% {pos}/{len} {per_sec} {elapsed_precise}")
            .progress_chars("|| "),
    );
    run_n_games(cfg, policy, rng, buffer, num_games_in_main_thread, bar);
}

fn run_n_games<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<G, N>,
    num_games: usize,
    progress_bar: ProgressBar,
) {
    let mut cached_policy = PolicyWithCache::with_capacity(100 * cfg.games_per_train, policy);

    for _ in 0..num_games {
        buffer.new_game();
        run_game(cfg, &mut cached_policy, rng, buffer);
        progress_bar.inc(1);
    }
    progress_bar.finish();
}

struct StateInfo {
    turn: usize,
    t: f32,
    q: f32,
    z: f32,
}

impl StateInfo {
    fn q(turn: usize, q: f32) -> Self {
        Self {
            turn,
            t: 0.0,
            q,
            z: 0.0,
        }
    }
}

fn run_game<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<G, N>,
) {
    let mut game = G::new();
    let mut solution = None;
    let mut search_policy = [0.0; N];
    let mut num_turns = 0;
    let mut state_infos = Vec::with_capacity(100);

    while solution.is_none() {
        let mut mcts = MCTS::with_capacity(
            cfg.num_explores + 1,
            cfg.learner_mcts_cfg,
            policy,
            game.clone(),
        );

        // add in noise to search process
        add_noise(cfg, &mut mcts, rng);

        // explore
        mcts.explore_n(cfg.num_explores);

        // store in buffer
        mcts.target_policy(&mut search_policy);
        buffer.add(&game, &search_policy, 0.0);
        state_infos.push(StateInfo::q(num_turns + 1, mcts.target_q()));

        // pick action
        let action = sample_action(cfg, &mut mcts, &game, &search_policy, rng, num_turns);
        solution = mcts.solution(&action);

        let is_over = game.step(&action);
        if is_over {
            solution = Some(game.reward(game.player()).into());
        } else if !cfg.stop_games_when_solved {
            solution = None;
        }
        num_turns += 1;
    }

    fill_state_info(&mut state_infos, solution.unwrap().reversed());
    store_rewards(cfg, buffer, &state_infos);
}

fn add_noise<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &LearningConfig,
    mcts: &mut MCTS<G, P, N>,
    rng: &mut R,
) {
    match cfg.noise {
        RolloutNoise::None => {}
        RolloutNoise::Equal { weight } => {
            mcts.add_equalizing_noise(weight);
        }
        RolloutNoise::Dirichlet { alpha, weight } => {
            mcts.add_dirichlet_noise(rng, alpha, weight);
        }
    }
}

fn sample_action<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &LearningConfig,
    mcts: &mut MCTS<G, P, N>,
    game: &G,
    search_policy: &[f32],
    rng: &mut R,
    num_turns: usize,
) -> G::Action {
    let best = mcts.best_action();
    let solution = mcts.solution(&best);
    let action = if num_turns < cfg.num_random_actions {
        let n = rng.gen_range(0..game.iter_actions().count() as u8) as usize;
        game.iter_actions().nth(n).unwrap()
    } else if num_turns < cfg.sample_action_until
        && (solution.is_none() || !cfg.stop_games_when_solved)
    {
        let dist = WeightedIndex::new(search_policy).unwrap();
        let choice = dist.sample(rng);
        // assert!(search_policy[choice] > 0.0);
        G::Action::from(choice)
    } else {
        best
    };
    action
}

fn fill_state_info(state_infos: &mut Vec<StateInfo>, mut outcome: Outcome) {
    let num_turns = state_infos.len();
    for state_value in state_infos.iter_mut().rev() {
        state_value.z = outcome.value();
        state_value.t = state_value.turn as f32 / num_turns as f32;
        outcome = outcome.reversed();
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
            ValueTarget::QtoZ { from, to } => {
                let p = (1.0 - state.t) * from + state.t * to;
                state.q * (1.0 - p) + state.z * p
            }
        };
    }
}
