mod connect4;
mod env;
mod mcts;
mod model;
mod runner;

use crate::connect4::{Connect4, PlayerId};
use crate::env::Env;
use crate::mcts::MCTS;
use crate::model::ConvNet;
use crate::runner::{gather_experience, run_game, RunConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::default::Default;
use tch::{
    nn::{Adam, OptimizerConfig, VarStore},
    Device,
};

fn main() {
    let seed = 0u64;
    tch::manual_seed(seed as i64);

    let mut vs = VarStore::new(Device::Cpu);
    let mut policy = ConvNet::new::<Connect4>(&vs.root());
    let mut opt = Adam::default().build(&vs, 1e-3).unwrap();
    opt.set_weight_decay(1e-5);

    let cfg = RunConfig {
        seed,
        capacity: 1_000_000,
        num_explores: 1500,
        temperature: 1.0,
        kind: tch::Kind::Float,
        device: tch::Device::Cpu,
    };
    let ts = gather_experience::<Connect4, ConvNet>(&cfg, &policy, 1_000);

    for (i, t) in ts.iter().enumerate() {
        println!("t {} {:?}", i, t);
    }
}
