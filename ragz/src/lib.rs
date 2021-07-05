mod data;
pub mod env;
mod mcts;
pub mod policies;
pub mod prelude;
mod runner;
mod utils;

use crate::data::*;
use crate::env::*;
use crate::policies::*;
use crate::runner::*;
pub use crate::runner::{RolloutConfig, ValueTarget};
pub use crate::utils::train_dir;
use crate::utils::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::default::Default;
use tch::{
    kind::Kind,
    nn::{Adam, OptimizerConfig, VarStore},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainConfig {
    pub lr: f64,
    pub weight_decay: f64,
    pub num_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: i64,
    pub seed: u64,
    pub logs: std::path::PathBuf,
}

pub fn evaluator<E: Env<N>, P: Policy<E, N> + NNPolicy<E, N>, const N: usize>(
    train_cfg: TrainConfig,
    rollout_cfg: RolloutConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let models_dir = train_cfg.logs.join("models");
    let pgn_path = train_cfg.logs.join("results.pgn");
    let mut pgn = std::fs::File::create(&pgn_path)?;
    let _guard = tch::no_grad_guard();
    let first_player = E::new().player();
    let all_explores = [100, 200, 400, 800, 1600, 3200, 6400, 12800];

    for i_iter in 0..train_cfg.num_iterations + 1 {
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

        let result = eval_against_random(&rollout_cfg, &mut policy, first_player);
        add_pgn_result(&mut pgn, &name, &String::from("Random"), result)?;
        let result = eval_against_random(&rollout_cfg, &mut policy, first_player.next());
        add_pgn_result(&mut pgn, &String::from("Random"), &name, result)?;

        for &explores in &all_explores {
            for seed in 0..10 {
                let result = eval_against_vanilla_mcts(
                    &rollout_cfg,
                    &mut policy,
                    first_player,
                    explores,
                    seed,
                );
                add_pgn_result(&mut pgn, &name, &format!("VanillaMCTS{}", explores), result)?;
                let result = eval_against_vanilla_mcts(
                    &rollout_cfg,
                    &mut policy,
                    first_player.next(),
                    explores,
                    seed,
                );
                add_pgn_result(&mut pgn, &format!("VanillaMCTS{}", explores), &name, result)?;
            }
        }

        // update results
        calculate_ratings(&train_cfg.logs)?;
        plot_ratings(&train_cfg.logs)?;
    }

    Ok(())
}

pub fn trainer<E: Env<N>, P: Policy<E, N> + NNPolicy<E, N>, const N: usize>(
    train_cfg: &TrainConfig,
    rollout_cfg: &RolloutConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&train_cfg.logs)?;
    let models_dir = train_cfg.logs.join("models");

    std::fs::create_dir(&models_dir)?;
    save(&train_cfg.logs, "train_cfg.json", train_cfg)?;
    save(&train_cfg.logs, "rollout_cfg.json", rollout_cfg)?;
    save_str(&train_cfg.logs, "env_name", &E::NAME.into())?;
    save_str(&train_cfg.logs, "git_hash", &git_hash()?)?;
    save_str(&train_cfg.logs, "git_diff.patch", &git_diff()?)?;

    tch::manual_seed(train_cfg.seed as i64);
    let mut rng = StdRng::seed_from_u64(train_cfg.seed);

    let vs = VarStore::new(tch::Device::Cpu);
    let mut policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, train_cfg.lr)?;
    opt.set_weight_decay(train_cfg.weight_decay);

    let mut dims = E::get_state_dims();

    vs.save(models_dir.join(String::from("model_0.ot")))?;

    let mut buffer = ReplayBuffer::<E, N>::new(rollout_cfg.buffer_size);
    fill_buffer(rollout_cfg, &mut rng, &mut buffer);

    for i_iter in 0..train_cfg.num_iterations {
        // gather data
        {
            let _guard = tch::no_grad_guard();
            gather_experience::<E, P, StdRng, N>(rollout_cfg, &mut policy, &mut rng, &mut buffer);
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

        // train
        for _i_epoch in 0..train_cfg.num_epochs {
            let sampler =
                BatchRandSampler::new(&states, &target_pis, &target_vs, train_cfg.batch_size, true);

            let mut epoch_loss = [0.0, 0.0];
            for (state, target_pi, target_v) in sampler {
                assert_eq!(state.size()[0], train_cfg.batch_size);

                let (logits, v) = policy.forward(&state);
                assert_eq!(logits.size(), target_pi.size());
                assert_eq!(v.size(), target_v.size());

                let log_pi = logits.log_softmax(-1, Kind::Float);
                let zeros = tch::Tensor::zeros_like(&target_pi);
                let legal_log_pi = log_pi.where1(&target_pi.greater1(&zeros), &zeros);

                // let pi_loss = (-legal_log_pi * target_pi)
                //     .sum1(&[-1], true, Kind::Float)
                //     .mean(Kind::Float);
                let pi_loss = (legal_log_pi * -target_pi).mean(Kind::Float);
                let v_loss = (v - target_v).square().mean(Kind::Float);

                let loss = &pi_loss + &v_loss;
                opt.backward_step(&loss);

                epoch_loss[0] += f32::from(&pi_loss);
                epoch_loss[1] += f32::from(&v_loss);
            }
            epoch_loss[0] *= (train_cfg.batch_size as f32) / (dims[0] as f32);
            epoch_loss[1] *= (train_cfg.batch_size as f32) / (dims[0] as f32);
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
