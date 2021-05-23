mod connect4;
mod data;
mod env;
mod mcts;
mod model;
mod runner;

use crate::connect4::Connect4;
use crate::data::BatchRandSampler;
use crate::model::ConvNet;
use crate::runner::{gather_experience, RunConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::default::Default;
use tch::nn::{Adam, OptimizerConfig, VarStore};

#[derive(Debug)]
struct TrainConfig {
    pub lr: f64,
    pub weight_decay: f64,
    pub num_epochs: usize,
    pub batch_size: i64,
}

fn main() {
    let seed = 0u64;
    tch::manual_seed(seed as i64);
    let mut rng = StdRng::seed_from_u64(seed);

    let train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 1e-5,
        num_epochs: 1,
        batch_size: 128,
    };

    let rollout_cfg = RunConfig {
        capacity: 1_000_000,
        num_explores: 400,
        temperature: 1.0,
        kind: tch::Kind::Float,
        device: tch::Device::cuda_if_available(),
        sample_action: true,
        steps_per_epoch: (train_cfg.batch_size * 1) as usize,
    };

    let vs = VarStore::new(rollout_cfg.device);
    let policy = ConvNet::new::<Connect4>(&vs);
    let mut opt = Adam::default().build(&vs, train_cfg.lr).unwrap();
    opt.set_weight_decay(train_cfg.weight_decay);

    println!("{:?}", train_cfg);
    println!("{:?}", rollout_cfg);
    for i_epoch in 0..train_cfg.num_epochs {
        let (states, target_pis, target_vs) =
            gather_experience::<Connect4, ConvNet, StdRng>(&rollout_cfg, &policy, &mut rng);
        let mut train_loss = 0f32;
        let sampler = BatchRandSampler::new(
            states,
            target_pis,
            target_vs,
            train_cfg.batch_size,
            true,
            rollout_cfg.device,
        );

        for (state, target_pi, target_v) in sampler {
            println!("Batch size {:?}", state.size()[0]);
            let (pi, v) = policy.forward(&state);
            let pi_loss = -(target_pi * pi.log()).mean(rollout_cfg.kind);
            let v_loss = v.mse_loss(&target_v, tch::Reduction::Mean);

            let loss = pi_loss + v_loss;
            opt.backward_step(&loss);

            train_loss += f32::from(&loss);
        }

        println!("Epoch {:?} - loss={}", i_epoch, train_loss);
    }
}
