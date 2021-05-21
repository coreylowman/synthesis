extern crate rand;

mod connect4;
mod env;

use crate::rand::rngs::StdRng;
use crate::rand::SeedableRng;
use connect4::Connect4;
use env::Env;
// use tch::nn::{Adam, ModuleT, OptimizerConfig, VarStore};
// use tch::vision::dataset::Dataset;
// use tch::TrainableCModule;
// use tch::{CModule, Device};

fn main() {
    let mut game = Connect4::new();
    let mut rng = StdRng::seed_from_u64(0);
    game.print();
    loop {
        let action = game.get_random_action(&mut rng);
        println!("-----");
        println!("Applying action {:?}", action);
        let is_over = game.step(&action);
        game.print();
        if is_over {
            break;
        }
    }
    // let device = Device::Cpu;
    // let vs = VarStore::new(device);
    // let mut trainable = TrainableCModule::load("model.pt", vs.root()).unwrap();
}
