mod connect4;
mod data;
mod env;
mod mcts;
mod model;
mod runner;

use crate::connect4::Connect4;
use crate::data::{tensor, BatchRandSampler};
use crate::env::Env;
use crate::mcts::Policy;
use crate::model::{ConvNet, UniformRandomPolicy};
use crate::runner::{gather_experience, RunConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use runner::run_game;
use std::default::Default;
use std::ffi::c_void;
use std::time::Instant;
use tch::{
    kind::{self, Element},
    nn::{Adam, OptimizerConfig, VarStore},
    Kind, Tensor,
};

#[derive(Debug)]
struct TrainConfig {
    pub lr: f64,
    pub weight_decay: f64,
    pub num_epochs: usize,
    pub batch_size: i64,
    pub device: tch::Device,
    pub kind: tch::Kind,
}

fn train(rng: &mut StdRng, rollout_cfg: &RunConfig, train_cfg: &TrainConfig) {
    let vs = VarStore::new(train_cfg.device);
    let policy = ConvNet::new::<Connect4>(&vs);
    let mut opt = Adam::default().build(&vs, train_cfg.lr).unwrap();
    opt.set_weight_decay(train_cfg.weight_decay);

    let mut state_dims = Connect4::get_state_dims();
    state_dims.insert(0, -1);
    let mut action_dims = [-1, Connect4::MAX_NUM_ACTIONS as i64];
    let mut value_dims = [-1, 1];

    println!("{:?}", train_cfg);
    println!("{:?}", rollout_cfg);
    for i_epoch in 0..train_cfg.num_epochs {
        // gather data
        let (states, target_pis, target_vs) = {
            let guard = tch::no_grad_guard();
            let items = gather_experience::<Connect4, ConvNet, StdRng>(rollout_cfg, &policy, rng);
            drop(guard);
            items
        };

        // convert to tensors
        state_dims[0] = target_vs.len() as i64;
        action_dims[0] = target_vs.len() as i64;
        value_dims[0] = target_vs.len() as i64;
        let states = tensor(&states, &state_dims, train_cfg.kind);
        let target_pis = tensor(&target_pis, &action_dims, train_cfg.kind);
        let target_vs = tensor(&target_vs, &value_dims, train_cfg.kind);
        println!("{:?} {:?} {:?}", states, target_pis, target_vs);

        // construct sampler
        let sampler = BatchRandSampler::new(
            states,
            target_pis,
            target_vs,
            train_cfg.batch_size,
            true,
            train_cfg.device,
        );

        // train
        let mut train_loss = 0f32;
        for (state, target_pi, target_v) in sampler {
            let (pi, v) = policy.forward(&state);

            let pi_loss = -(target_pi * pi.log()).mean(train_cfg.kind);
            let v_loss = v.mse_loss(&target_v, tch::Reduction::Mean);

            let loss = pi_loss + v_loss;
            opt.backward_step(&loss);

            train_loss += f32::from(&loss);
        }

        println!("Epoch {:?} - loss={}", i_epoch, train_loss);
    }
}

fn bench(rng: &mut StdRng, rollout_cfg: &RunConfig) {
    // let policy = UniformRandomPolicy;
    let vs = VarStore::new(tch::Device::Cpu);
    let policy = ConvNet::new::<Connect4>(&vs);
    let game = Connect4::new();
    loop {
        let start = Instant::now();
        // run_game::<Connect4, UniformRandomPolicy, StdRng>(rollout_cfg, &policy, rng);
        let (states, pis, vs) =
            gather_experience::<Connect4, ConvNet, StdRng>(rollout_cfg, &policy, rng);
        println!("{:?}", vs.len());
        // for _ in 0..10000 {
        //     let x = game.state();
        // }
        // let xs = game.state();
        // let t = tensor(&xs, &[1, 1, 6, 7], tch::Kind::Float);
        // for _ in 0..1000 {
        //     // let (pi, v) = policy.eval(&game);
        //     let (policy, value) = policy.forward(&t);
        // }

        println!("{:?}", start.elapsed());
    }
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
        kind: tch::Kind::Float,
        device: tch::Device::Cpu,
    };

    let rollout_cfg = RunConfig {
        capacity: 100_000,
        num_explores: 100,
        temperature: 1.0,
        sample_action: true,
        steps_per_epoch: (train_cfg.batch_size * 10) as usize,
    };

    train(&mut rng, &rollout_cfg, &train_cfg);
    // let guard = tch::no_grad_guard();
    // bench(&mut rng, &rollout_cfg);
}
