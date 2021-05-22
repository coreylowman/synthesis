mod connect4;
mod env;
mod mcts;
mod model;
mod runner;

use crate::connect4::Connect4;
use crate::model::ConvNet;
use crate::runner::{gather_experience, RunConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::default::Default;
use tch::nn::{Adam, OptimizerConfig, VarStore};

fn main() {
    let seed = 0u64;
    tch::manual_seed(seed as i64);
    let mut rng = StdRng::seed_from_u64(seed);

    let train_cfg = RunConfig {
        capacity: 1_000_000,
        num_explores: 400,
        temperature: 1.0,
        kind: tch::Kind::Float,
        device: tch::Device::cuda_if_available(),
        sample_action: true,
    };

    let mut vs = VarStore::new(train_cfg.device);
    let mut policy = ConvNet::new::<Connect4>(&vs);
    let mut opt = Adam::default().build(&vs, 1e-3).unwrap();
    opt.set_weight_decay(1e-5);

    println!("{:?}", train_cfg);
    let ts = gather_experience::<Connect4, ConvNet, StdRng>(&train_cfg, &policy, &mut rng, 100);

    for (i, t) in ts.iter().enumerate() {
        println!("t {} {:?}", i, t);
    }
}
