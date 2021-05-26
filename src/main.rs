mod connect4;
mod data;
mod env;
mod mcts;
mod model;
mod runner;

use crate::connect4::Connect4;
use crate::data::{tensor, BatchRandSampler};
use crate::env::Env;
use crate::mcts::{Node, Policy};
use crate::model::{ConvNet, NNPolicy};
use crate::runner::{eval, gather_experience, EvaluationConfig, RolloutConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::default::Default;
use tch::nn::{Adam, OptimizerConfig, VarStore};

#[derive(Debug)]
struct TrainConfig {
    pub lr: f64,
    pub weight_decay: f64,
    pub num_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: i64,
    pub device: tch::Device,
    pub kind: tch::Kind,
}

fn train<E: Env, P: Policy<E> + NNPolicy<E>>(
    rng: &mut StdRng,
    rollout_cfg: &RolloutConfig,
    train_cfg: &TrainConfig,
    evaluation_cfg: &EvaluationConfig,
) {
    let vs = VarStore::new(train_cfg.device);
    let mut policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, train_cfg.lr).unwrap();
    opt.set_weight_decay(train_cfg.weight_decay);

    let mut state_dims = E::get_state_dims();
    state_dims.insert(0, -1);
    let mut action_dims = [-1, E::MAX_NUM_ACTIONS as i64];
    let mut value_dims = [-1, 1];

    println!("{:?}", rollout_cfg);
    println!("{:?}", train_cfg);
    println!("{:?}", evaluation_cfg);

    let win_pct = {
        let _guard = tch::no_grad_guard();
        eval(evaluation_cfg, rng, &mut policy)
    };
    println!("Init win pct {:.3}%", win_pct * 100.0);

    for i_iter in 0..train_cfg.num_iterations {
        println!("Iteration {}...", i_iter);

        // gather data
        let (states, target_pis, target_vs) = {
            let _guard = tch::no_grad_guard();
            gather_experience::<E, P, StdRng>(rollout_cfg, &mut policy, rng)
        };

        // convert to tensors
        state_dims[0] = target_vs.len() as i64;
        action_dims[0] = target_vs.len() as i64;
        value_dims[0] = target_vs.len() as i64;
        let states = tensor(&states, &state_dims, train_cfg.kind);
        let target_pis = tensor(&target_pis, &action_dims, train_cfg.kind);
        let target_vs = tensor(&target_vs, &value_dims, train_cfg.kind);

        for i_epoch in 0..train_cfg.num_epochs {
            let sampler =
                BatchRandSampler::new(&states, &target_pis, &target_vs, train_cfg.batch_size, true);

            // train
            let mut epoch_loss = 0f32;
            for (state, target_pi, target_v) in sampler {
                let (pi, v) = policy.forward(&state);

                let pi_loss = -(target_pi * pi.log()).mean(train_cfg.kind);
                let v_loss = v.mse_loss(&target_v, tch::Reduction::Mean);

                let loss = pi_loss + v_loss;
                opt.backward_step(&loss);

                epoch_loss += f32::from(&loss);
            }
            println!("\tEpoch {} loss={}", i_epoch, epoch_loss);
        }

        let win_pct = {
            let _guard = tch::no_grad_guard();
            eval(evaluation_cfg, rng, &mut policy)
        };
        println!("Iteration {} strength={:.3}%.", i_iter, win_pct * 100.0);
    }
}

fn main() {
    let seed = 0u64;
    tch::manual_seed(seed as i64);
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Connect4 {:?}", std::mem::size_of::<Connect4>());
    println!("Box<Connect4> {:?}", std::mem::size_of::<Box<Connect4>>());
    println!(
        "Connect4::ActionIterator {:?}",
        std::mem::size_of::<<Connect4 as Env>::ActionIterator>()
    );
    println!("Node<Connect4> {:?}", std::mem::size_of::<Node<Connect4>>());

    let train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 1e-5,
        num_iterations: 10,
        num_epochs: 8,
        batch_size: 128,
        kind: tch::Kind::Float,
        device: tch::Device::Cpu,
    };

    let rollout_cfg = RolloutConfig {
        capacity: 100_000,
        num_explores: 100,
        temperature: 1.0,
        sample_action: true,
        steps_per_epoch: (train_cfg.batch_size * 10) as usize,
    };

    let evaluation_cfg = EvaluationConfig {
        capacity: rollout_cfg.capacity,
        num_explores: rollout_cfg.num_explores,
        num_games: 100,
    };

    train::<Connect4, ConvNet<Connect4>>(&mut rng, &rollout_cfg, &train_cfg, &evaluation_cfg);
}
