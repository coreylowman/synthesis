mod connect4;
mod env;
mod mcts;
mod model;
mod runner;

use connect4::{Connect4, PlayerId};
use env::Env;
use mcts::{Policy, MCTS};
use model::ConvNet;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::default::Default;
use tch::nn::{Adam, Conv, Module, OptimizerConfig, Sequential, VarStore};
use tch::vision::dataset::Dataset;
use tch::TrainableCModule;
use tch::{nn, Device};

fn main() {
    let mut vs = VarStore::new(Device::Cpu);
    let mut policy = ConvNet::new(&vs.root());

    let mut game = Connect4::new();
    let mut rng = StdRng::seed_from_u64(1);
    let mut mcts = MCTS::<Connect4, ConvNet>::with_capacity(2_500_000, 0, policy);
    game.print();

    mcts.add_root();
    // loop {
    //     // let action = if game.player() == PlayerId::Red {
    //     //     mcts.explore_n(1500);
    //     //     mcts.best_action()
    //     // } else {
    //     let action = game.get_random_action(&mut rng);
    //     // };
    //     // mcts.step_action(&action);
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
