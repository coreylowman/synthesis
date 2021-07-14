pub mod config;
mod data;
pub mod game;
mod mcts;
pub mod policies;
pub mod prelude;
mod runner;
mod utils;

use crate::config::*;
use crate::data::*;
use crate::game::*;
use crate::policies::*;
use crate::runner::*;
pub use crate::utils::train_dir;
use crate::utils::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
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

pub fn evaluator<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: &LearningConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let models_dir = cfg.logs.join("models");
    let pgn_path = cfg.logs.join("results.pgn");
    let mut pgn = std::fs::File::create(&pgn_path)?;
    let _guard = tch::no_grad_guard();
    let first_player = G::new().player();
    let all_explores = [100, 200, 400, 800, 1600, 3200, 6400, 12800];

    for i in 0..all_explores.len() {
        for j in 0..all_explores.len() {
            if i == j {
                continue;
            }
            for seed in 0..10 {
                add_pgn_result(
                    &mut pgn,
                    &format!("VanillaMCTS{}", all_explores[i]),
                    &format!("VanillaMCTS{}", all_explores[j]),
                    mcts_vs_mcts::<G, N>(
                        &cfg,
                        first_player,
                        all_explores[i],
                        all_explores[j],
                        seed,
                    ),
                )?;
            }
        }
    }

    for i_iter in 0..cfg.num_iterations + 1 {
        // wait for model to exist;
        let name = format!("model_{}.ot", i_iter);
        while !models_dir.join(&name).exists() {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        // wait an extra second to be sure data is there
        std::thread::sleep(std::time::Duration::from_secs(1));

        // load model
        let mut vs = VarStore::new(tch::Device::Cpu);
        let policy = P::new(&vs);
        vs.load(models_dir.join(&name))?;
        let mut policy = OwnedPolicyWithCache {
            policy,
            cache: HashMap::with_capacity(100_000),
        };

        let result = eval_against_random(&cfg, &mut policy, first_player);
        add_pgn_result(&mut pgn, &name, &String::from("Random"), result)?;
        let result = eval_against_random(&cfg, &mut policy, first_player.next());
        add_pgn_result(&mut pgn, &String::from("Random"), &name, result)?;

        for &explores in &all_explores {
            let op_name = format!("VanillaMCTS{}", explores);
            for seed in 0..10 {
                let result =
                    eval_against_vanilla_mcts(&cfg, &mut policy, first_player, explores, seed);
                add_pgn_result(&mut pgn, &name, &op_name, result)?;
                let result = eval_against_vanilla_mcts(
                    &cfg,
                    &mut policy,
                    first_player.next(),
                    explores,
                    seed,
                );
                add_pgn_result(&mut pgn, &op_name, &name, result)?;
            }
        }

        // update results
        calculate_ratings(&cfg.logs)?;
        plot_ratings(&cfg.logs)?;
    }

    Ok(())
}
