mod envs;
mod policy_impls;

use crate::envs::*;
use crate::policy_impls::*;
use ragz::prelude::*;
use ragz::{evaluator, train_dir, trainer, RolloutConfig, TrainConfig, ValueTarget};

fn run<E: Env<N>, P: Policy<E, N> + NNPolicy<E, N>, const N: usize>(
) -> Result<(), Box<dyn std::error::Error>> {
    let train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 0.0,
        num_iterations: 200,
        num_epochs: 20,
        batch_size: 64,
        seed: 0,
        logs: train_dir("./_logs", E::NAME)?,
    };

    // TODO try 1600 explores and see if still stuck at 400
    // TODO try noisy_explore with noise_weight of 0.0625
    // TODO try c_puct 1.0
    // TODO why does num_random_actions > 0 affect things so much?

    let rollout_cfg = RolloutConfig {
        buffer_size: 256_000,
        games_to_keep: 8000,
        games_per_train: 2000,
        num_explores: 800,
        num_random_actions: 1,
        sample_action_until: 25,
        alpha: 10.0 / (N as f32),
        noisy_explore: true,
        noise_weight: 0.25,
        c_puct: 1.0,
        solve: true,
        fpu: f32::INFINITY,
        value_target: ValueTarget::Interpolate,
    };

    tch::set_num_threads(2);
    tch::set_num_interop_threads(2);

    let eval_train_cfg = train_cfg.clone();
    let eval_rollout_cfg = rollout_cfg.clone();
    let eval_handle =
        std::thread::spawn(move || evaluator::<E, P, N>(eval_train_cfg, eval_rollout_cfg).unwrap());
    trainer::<E, P, N>(&train_cfg, &rollout_cfg)?;
    eval_handle.join().unwrap();
    Ok(())
}

fn main() {
    run::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>().unwrap()
}
