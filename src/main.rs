mod connect4;
mod env;
mod mcts;
mod model;
mod runner;

use crate::connect4::{Connect4, PlayerId};
use crate::env::Env;
use crate::mcts::MCTS;
use crate::model::ConvNet;
use crate::runner::{run_game, RunConfig};
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
    let ts = run_game::<Connect4, ConvNet>(&cfg, policy);

    for (i, t) in ts.iter().enumerate() {
        println!("t {} {:?}", i, t);
    }

    // let mut rng = StdRng::seed_from_u64(seed);
    // let mut game = Connect4::new();
    // let mut mcts = MCTS::<Connect4, ConvNet>::with_capacity(2_500_000, 0, policy);
    // game.print();

    // loop {
    //     let action = if game.player() == PlayerId::Red {
    //         mcts.explore_n(1500);
    //         mcts.best_action()
    //     } else {
    //         game.get_random_action(&mut rng)
    //     };
    //     mcts.step_action(&action);
    //     println!("-----");
    //     println!("Applying action {:?}", action);
    //     let is_over = game.step(&action);
    //     game.print();
    //     // game.state().print();
    //     if is_over {
    //         break;
    //     }
    // }
}
