mod connect4;
mod env;
mod mcts;
mod model;
mod runner;

use connect4::{Connect4, PlayerId};
use env::Env;
use mcts::MCTS;
use model::NNPolicy;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::default::Default;
use tch::nn::{Adam, Module, OptimizerConfig, Sequential, VarStore};
use tch::{nn, Device};
// use tch::vision::dataset::Dataset;
// use tch::TrainableCModule;
// use tch::{CModule, Device};

fn model(p: &nn::Path) -> Sequential {
    let stride = |s| nn::ConvConfig {
        stride: s,
        ..Default::default()
    };
    let seq = nn::seq()
        .add(nn::conv2d(p / "c1", 2, 32, 8, stride(4)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
        .add_fn(|xs| xs.relu().flat_view())
        .add(nn::linear(p / "l1", 3136, 512, Default::default()))
        .add_fn(|xs| xs.relu());
    seq
}

fn main() {
    let mut vs = VarStore::new(Device::Cpu);
    let mut model = model(&vs.root());
    let policy = NNPolicy { model };

    let mut game = Connect4::new();
    let mut rng = StdRng::seed_from_u64(1);
    let mut mcts = MCTS::<Connect4, NNPolicy<Sequential>>::with_capacity(2_500_000, 0, policy);
    game.print();
    loop {
        let action = if game.player() == PlayerId::Red {
            mcts.explore_n(1500);
            mcts.best_action()
        } else {
            game.get_random_action(&mut rng)
        };
        mcts.step_action(&action);
        println!("-----");
        println!("Applying action {:?}", action);
        let is_over = game.step(&action);
        game.print();
        // game.state().print();
        if is_over {
            break;
        }
    }
    // let device = Device::Cpu;
    // let vs = VarStore::new(device);
    // let mut trainable = TrainableCModule::load("model.pt", vs.root()).unwrap();
}
