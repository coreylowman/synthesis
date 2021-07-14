mod connect4;
mod policies;

use crate::connect4::Connect4;
use crate::policies::*;
use synthesis::prelude::*;

fn learn<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
) -> Result<(), Box<dyn std::error::Error>> {
    // TODO try 1600 explores and see if still stuck at 400
    // TODO try noisy_explore with noise_weight of 0.0625
    // TODO try c_puct 1.0
    // TODO why does num_random_actions > 0 affect things so much?

    let cfg = LearningConfig {
        seed: 0,
        logs: train_dir("./_logs", G::NAME)?,

        lr: 1e-3,
        weight_decay: 0.0,
        num_iterations: 200,
        num_epochs: 20,
        value_target: ValueTarget::Interpolate,

        batch_size: 64,
        buffer_size: 256_000,
        games_to_keep: 8000,
        games_per_train: 2000,

        num_explores: 800,
        num_random_actions: 1,
        sample_action_until: 25,
        alpha: 10.0 / (N as f32),
        noisy_explore: true,
        noise_weight: 0.25,
        c_puct: 1.0,
        solve: true,
        fpu: f32::INFINITY,
    };

    tch::set_num_threads(2);
    tch::set_num_interop_threads(2);

    let eval_cfg = cfg.clone();
    let eval_handle = std::thread::spawn(move || evaluator::<G, P, N>(&eval_cfg).unwrap());
    learner::<G, P, N>(&cfg)?;
    eval_handle.join().unwrap();
    Ok(())
}

fn main() {
    learn::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>().unwrap()
}
