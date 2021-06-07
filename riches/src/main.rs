mod envs;
mod policy_impls;

use crate::envs::*;
use crate::policy_impls::*;
use ragz::prelude::*;
use ragz::{train, RolloutConfig, TrainConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 1e-5,
        num_iterations: 200,
        num_epochs: 2,
        batch_size: 256,
        buffer_size: 16_000,
        seed: 0,
        logs: "./_logs",
    };

    let rollout_cfg = RolloutConfig {
        capacity: 100_000,
        num_explores: 800,
        temperature: 1.0,
        sample_action: true,
        steps: 3_200,
        alpha: 1.0,
        noisy_explore: true,
        c_puct: 4.0,
    };

    train::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>(&train_cfg, &rollout_cfg)
}
