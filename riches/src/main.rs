mod envs;
mod policy_impls;

use crate::envs::*;
use crate::policy_impls::*;
use ragz::prelude::*;
use ragz::{evaluator, train_dir, trainer, RolloutConfig, TrainConfig};

fn run<E: Env<N>, P: Policy<E, N> + NNPolicy<E, N>, const N: usize>(
) -> Result<(), Box<dyn std::error::Error>> {
    let train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 1e-4,
        num_iterations: 200,
        num_epochs: 10,
        batch_size: 2056,
        seed: 0,
        logs: train_dir("./_logs", E::NAME)?,
    };

    let rollout_cfg = RolloutConfig {
        buffer_size: 128_000,
        num_explores: 800,
        sample_action_until: 25,
        steps: 25_600,
        alpha: 1.0 / (N as f32),
        noisy_explore: true,
        noise_weight: 0.25,
        c_puct: 2.0,
    };

    let eval_train_cfg = train_cfg.clone();
    let eval_rollout_cfg = rollout_cfg.clone();
    let eval_handle =
        std::thread::spawn(move || evaluator::<E, P, N>(eval_train_cfg, eval_rollout_cfg).unwrap());
    trainer::<E, P, N>(&train_cfg, &rollout_cfg)?;
    eval_handle.join().unwrap();
    Ok(())
}

fn main() {
    // run::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>().unwrap()

    let eval_train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 1e-4,
        num_iterations: 200,
        num_epochs: 10,
        batch_size: 2056,
        seed: 0,
        logs: std::path::Path::new("./_logs")
            .join("Connect4")
            .join("06-12-2021T15-46-51Z"),
    };

    let eval_rollout_cfg = RolloutConfig {
        buffer_size: 128_000,
        num_explores: 800,
        sample_action_until: 25,
        steps: 25_600,
        alpha: 1.0 / (Connect4::MAX_NUM_ACTIONS as f32),
        noisy_explore: true,
        noise_weight: 0.25,
        c_puct: 2.0,
    };

    evaluator::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>(
        eval_train_cfg,
        eval_rollout_cfg,
    )
    .unwrap()
}
