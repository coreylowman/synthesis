use crate::config::{LearningConfig, RolloutConfig, ValueTarget};
use crate::data::*;
use crate::game::{Game, Outcome};
use crate::mcts::MCTS;
use crate::policies::{NNPolicy, Policy, PolicyWithCache};
use crate::utils::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand::{distributions::Distribution, distributions::WeightedIndex};
use std::default::Default;
use tch::{
    kind::Kind,
    nn::{Adam, OptimizerConfig, VarStore},
};

pub fn alpha_zero<G: 'static + Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: &LearningConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // set up directory structure
    std::fs::create_dir_all(&cfg.logs)?;
    let models_dir = cfg.logs.join("models");
    std::fs::create_dir(&models_dir)?;
    save_str(&cfg.logs, "env_name", &G::NAME.into())?;
    save_str(&cfg.logs, "git_hash", &git_hash()?)?;
    save_str(&cfg.logs, "git_diff.patch", &git_diff()?)?;

    // seed rngs
    tch::manual_seed(cfg.seed as i64);

    // init policy
    let vs = VarStore::new(tch::Device::Cpu);
    let policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, cfg.lr_schedule[0].1)?;
    if cfg.weight_decay > 0.0 {
        opt.set_weight_decay(cfg.weight_decay);
    }
    vs.save(models_dir.join(String::from("model_0.ot")))?;

    // init replay buffer
    let mut buffer = ReplayBuffer::new(256_000);

    // start learning!
    let mut dims = G::DIMS.to_owned();
    let batch_mean = 1.0 / (cfg.batch_size as f32);
    for i_iter in 0..cfg.num_iterations {
        // gather data
        {
            let _guard = tch::no_grad_guard();
            gather_experience::<G, P, N>(cfg, format!("model_{}.ot", i_iter), &mut buffer, i_iter);
        }

        // convert buffer data to tensors
        let dedup = buffer.deduplicate();
        println!("Dedup {} -> {} steps", buffer.vs.len(), dedup.vs.len());
        dims[0] = dedup.vs.len() as i64;
        let states = tensor(&dedup.states, &dims, Kind::Float);
        let target_pis = tensor(&dedup.pis, &[dims[0], N as i64], Kind::Float);
        let target_vs = tensor(&dedup.vs, &[dims[0], 3], Kind::Float);

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
                let (pi_logits, v_logits) = policy.forward(&state);

                let log_pi = pi_logits.log_softmax(-1, Kind::Float);
                let log_v = v_logits.log_softmax(-1, Kind::Float);
                let pi_loss = batch_mean * log_pi.kl_div(&target_pi, tch::Reduction::Sum, false);
                let v_loss = batch_mean * log_v.kl_div(&target_v, tch::Reduction::Sum, false);

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

        println!("Finished iteration {}", i_iter + 1);
        println!(
            "lifetime: {} steps / {} games ({:.3} steps/game)",
            buffer.total_steps(),
            buffer.total_games_played(),
            buffer.total_steps() as f32 / buffer.total_games_played() as f32,
        );
        println!(
            "buffer: {} steps / {} games ({:.3} steps/game)",
            buffer.curr_steps(),
            buffer.curr_games(),
            buffer.curr_steps() as f32 / buffer.curr_games() as f32,
        );
    }

    Ok(())
}

fn gather_experience<G: 'static + Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: &LearningConfig,
    policy_name: String,
    buffer: &mut ReplayBuffer<G, N>,
    seed: usize,
) {
    let mut games_to_schedule = cfg.games_per_train;
    let mut workers_left = cfg.rollout_cfg.num_workers + 1;
    let mut handles = Vec::with_capacity(workers_left);
    let multi_bar = MultiProgress::new();

    // create workers
    for i_worker in 0..cfg.rollout_cfg.num_workers + 1 {
        // create copies of data for this worker
        let worker_policy_name = policy_name.clone();
        let worker_cfg = cfg.clone();

        // calculate number of games this worker will run. this allows uneven number of games across workers
        let num_games = games_to_schedule / workers_left;
        let worker_bar = multi_bar.add(styled_progress_bar(num_games));
        let worker_seed = seed * (cfg.rollout_cfg.num_workers + 1) + i_worker;
        // spawn a worker
        handles.push(std::thread::spawn(move || {
            run_n_games::<G, P, N>(
                worker_cfg,
                worker_policy_name,
                num_games,
                worker_bar,
                worker_seed,
            )
        }));

        games_to_schedule -= num_games;
        workers_left -= 1;
    }

    // sanity check that all games are scheduled
    assert!(games_to_schedule == 0);
    assert!(workers_left == 0);

    // wait for workers to complete
    multi_bar.join().unwrap();

    // collect experience gathered into main buffer
    buffer.keep_last_n_games(cfg.games_to_keep - cfg.games_per_train);
    for handle in handles.drain(..) {
        let mut worker_buffer = handle.join().unwrap();
        buffer.extend(&mut worker_buffer);
    }
}

fn styled_progress_bar(n: usize) -> ProgressBar {
    let bar = ProgressBar::new(n as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} ({percent}%) | {eta} remaining | {elapsed_precise}")
            .progress_chars("|| "),
    );
    bar
}

fn run_n_games<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: LearningConfig,
    policy_name: String,
    num_games: usize,
    progress_bar: ProgressBar,
    seed: usize,
) -> ReplayBuffer<G, N> {
    let mut buffer = ReplayBuffer::new(G::MAX_TURNS * num_games);
    let mut rng = StdRng::seed_from_u64(seed as u64);

    // load the policy weights
    let mut vs = VarStore::new(tch::Device::Cpu);
    let mut policy = P::new(&vs);
    vs.load(cfg.logs.join("models").join(&policy_name)).unwrap();

    // create a cache for this policy, this speeds things up a lot, but takes memory
    let mut cached_policy =
        PolicyWithCache::with_capacity(G::MAX_TURNS * cfg.games_per_train, &mut policy);

    // run all the games
    for _ in 0..num_games {
        buffer.new_game();
        run_game(&cfg.rollout_cfg, &mut cached_policy, &mut rng, &mut buffer);
        progress_bar.inc(1);
    }
    progress_bar.finish();

    buffer
}

struct StateInfo {
    turn: usize,
    t: f32,
    q: [f32; 3],
    z: [f32; 3],
}

impl StateInfo {
    fn q(turn: usize, q: [f32; 3]) -> Self {
        Self {
            turn,
            t: 0.0,
            q,
            z: [0.0; 3],
        }
    }
}

fn run_game<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    policy: &mut P,
    rng: &mut R,
    buffer: &mut ReplayBuffer<G, N>,
) {
    let mut game = G::new();
    let mut solution = None;
    let mut search_policy = [0.0; N];
    let mut num_turns = 0;
    let mut state_infos = Vec::with_capacity(G::MAX_TURNS);

    while solution.is_none() {
        let mut mcts =
            MCTS::with_capacity(cfg.num_explores + 1, cfg.mcts_cfg, policy, game.clone());

        // explore
        mcts.explore_n(cfg.num_explores);

        // store in buffer
        mcts.target_policy(&mut search_policy);
        buffer.add(&game, &search_policy, [0.0; 3]);
        state_infos.push(StateInfo::q(num_turns + 1, mcts.target_q()));

        // pick action
        let action = sample_action(&cfg, &mut mcts, &game, &search_policy, rng, num_turns);
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
    store_rewards(&cfg, buffer, &state_infos);
}

fn sample_action<G: Game<N>, P: Policy<G, N>, R: Rng, const N: usize>(
    cfg: &RolloutConfig,
    mcts: &mut MCTS<G, P, N>,
    game: &G,
    search_policy: &[f32],
    rng: &mut R,
    num_turns: usize,
) -> G::Action {
    let best = mcts.best_action(cfg.action);
    let solution = mcts.solution(&best);
    let action = if num_turns < cfg.random_actions_until {
        let n = rng.gen_range(0..game.iter_actions().count() as u8) as usize;
        game.iter_actions().nth(n).unwrap()
    } else if num_turns < cfg.sample_actions_until
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
        state_value.z[match outcome {
            Outcome::Win(_) => 2,
            Outcome::Draw(_) => 1,
            Outcome::Lose(_) => 0,
        }] = 1.0;
        state_value.t = state_value.turn as f32 / num_turns as f32;
        outcome = outcome.reversed();
    }
}

fn store_rewards<G: Game<N>, const N: usize>(
    cfg: &RolloutConfig,
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
            ValueTarget::QZaverage { p } => {
                let mut value = [0.0; 3];
                for i in 0..3 {
                    value[i] = state.q[i] * p + state.z[i] * (1.0 - p);
                }
                value
            }
            ValueTarget::QtoZ { from, to } => {
                let p = (1.0 - state.t) * from + state.t * to;
                let mut value = [0.0; 3];
                for i in 0..3 {
                    value[i] = state.q[i] * (1.0 - p) + state.z[i] * p;
                }
                value
            }
        };
    }
}
