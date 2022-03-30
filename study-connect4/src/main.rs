mod connect4;
mod policies;

use rand::{distributions::Distribution, thread_rng, Rng};
use rand_distr::{Dirichlet, Normal};

use crate::connect4::Connect4;
use crate::policies::*;
use synthesis::prelude::*;

// FIXME: the action with most number of visits may not necessarily have the highest q value (either from correction, or just poor selection)
// FIXME: similarly the visit distribution may not be accurate due to value correction

fn learn<G: 'static + Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LearningConfig {
        seed: 0,                              // seed for rng & torch
        logs: train_dir("./_logs", G::NAME)?, // log directory
        num_iterations: 200,                  // number of training iterations to run

        lr_schedule: vec![(1, 1e-3), (10, 5e-4), (20, 1e-4)], // schedule for lr - first item in tuple is iteration #
        weight_decay: 0.0,                                    // L2 regularization for optimizer
        num_epochs: 8,      // number of full passes over training data per iteration
        batch_size: 1024,   // size of batches that epochs are split into
        policy_weight: 1.0, // scalar for policy loss
        value_weight: 1.0,  // scalar for value loss

        games_to_keep: 1000,   // number of games to keep in replay buffer
        games_per_train: 1000, // number of new games to add to replay buffer per training iteration

        rollout_cfg: RolloutConfig {
            num_workers: 6,                     // number of processes to use for running games
            random_actions_until: 1,            // last turn number to select random actions
            sample_actions_until: 30,           // last turn number to sample actions
            stop_games_when_solved: false,      // end games early if they are solved by MCTS
            action: ActionSelection::NumVisits, // the value to use for best action
            value_target: ValueTarget::Q,       // the target for NN value function
            sample_num_explores: || {
                // number of MCTS explores per turn
                // NOTE: sampled for each turn
                800
            },
            sample_exploration: || {
                // type of exploration to use (e.g. PUCT or UCT)
                // NOTE: sampled for each turn
                Exploration::PolynomialUct { c: 1.0 }
            },

            mcts_cfg: MCTSConfig {
                auto_extend: true, // expand until a node with > 1 child is reached
                exploration: Exploration::PolynomialUct { c: 1.0 },
                solver_cfg: SolverConfig {
                    solve: true,                // use MCTS Solver
                    remove_action_prob: true,   // set solved node's action_prob to 0.0
                    select_solved_nodes: false, // select nodes that are solved
                    correct_values: true,       // adjust previously backprop'd values
                },
                fpu: Fpu::Func(|parent_q| {
                    // exploit value of un-evaluated nodes
                    let dist = Normal::new(parent_q, 0.1).unwrap();
                    dist.sample(&mut thread_rng())
                }),
                root_policy_noise: Some(PolicyNoise {
                    weight: 0.25,
                    sample_fn: |num_children| {
                        let dist = Dirichlet::new_with_size(1.0, num_children).unwrap();
                        dist.sample(&mut thread_rng())
                    },
                }),
            },
        },
    };

    let eval_cfg = EvaluationConfig {
        logs: cfg.logs.clone(),

        policy_num_explores: 800,
        policy_action: ActionSelection::NumVisits,
        policy_mcts_cfg: MCTSConfig {
            fpu: Fpu::ParentQ,       // NOTE no noise during eval
            root_policy_noise: None, // NOTE no noise during eval
            exploration: Exploration::PolynomialUct { c: 1.0 },
            ..cfg.rollout_cfg.mcts_cfg
        },

        num_games_against_best_policies: 1,
        num_best_policies: 10,
        num_games_against_rollout: 5,
        rollout_num_explores: vec![800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800],
    };

    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);

    let eval_handle = std::thread::spawn(move || evaluator::<G, P, N>(&eval_cfg).unwrap());
    alpha_zero::<G, P, N>(&cfg)?;
    eval_handle.join().unwrap();
    Ok(())
}

fn main() {
    learn::<Connect4, Connect4Net, { Connect4::MAX_NUM_ACTIONS }>().unwrap()
}
