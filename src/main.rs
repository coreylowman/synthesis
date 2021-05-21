mod connect4;
mod env;
mod mcts;

use connect4::{Connect4, PlayerId};
use env::Env;
use mcts::MCTS;
use rand::rngs::StdRng;
use rand::SeedableRng;
// use tch::nn::{Adam, ModuleT, OptimizerConfig, VarStore};
// use tch::vision::dataset::Dataset;
// use tch::TrainableCModule;
// use tch::{CModule, Device};

fn main() {
    let mut game = Connect4::new();
    let mut rng = StdRng::seed_from_u64(1);
    let mut mcts = MCTS::<Connect4>::with_capacity(PlayerId::Red, 2_500_000, 0);
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
        game.state().print();
        if is_over {
            break;
        }
    }
    // let device = Device::Cpu;
    // let vs = VarStore::new(device);
    // let mut trainable = TrainableCModule::load("model.pt", vs.root()).unwrap();
}
