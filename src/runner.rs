use crate::env::Env;
use crate::mcts::{Policy, MCTS};
use tch::{self, nn, IndexOp, Tensor};

#[derive(Debug)]
pub struct Timestep {
    pub state: Tensor,
    pub policy: Tensor,
    pub value: f32,
}

#[derive(Debug)]
pub struct RunConfig {
    pub seed: u64,
    pub capacity: usize,
    pub num_explores: usize,
    pub temperature: f64,
    pub kind: tch::Kind,
    pub device: tch::Device,
}

fn extract_policy<E: Env + Clone, P: Policy<E>>(cfg: &RunConfig, mcts: &MCTS<E, P>) -> Tensor {
    let mut policy = Tensor::zeros(&[E::MAX_NUM_ACTIONS as i64], (cfg.kind, cfg.device));
    for (action, num_visits) in mcts.visit_counts() {
        let action_id: usize = action.into();
        let _ = policy
            .i(action_id as i64)
            .fill_((num_visits as f64).powf(1.0 / cfg.temperature));
    }
    policy /= policy.sum(cfg.kind);
    policy
}

pub fn run_game<E: Env + Clone, P: Policy<E>>(cfg: &RunConfig, policy: P) -> Vec<Timestep> {
    let mut mcts = MCTS::<E, P>::with_capacity(cfg.capacity, cfg.seed, policy);
    let mut ts: Vec<Timestep> = Vec::new();
    let mut game = E::new();
    let root_player = game.player();
    let mut is_over = false;
    while !is_over {
        mcts.explore_n(cfg.num_explores);

        // save timestep
        ts.push(Timestep {
            state: game.state(),
            policy: extract_policy(cfg, &mcts),
            value: 0.0,
        });

        let action = mcts.best_action();
        mcts.step_action(&action);

        println!("-----");
        println!("Applying action {:?}", action);
        is_over = game.step(&action);
        game.print();
    }

    let mut r = -game.reward(game.player());
    for t in ts.iter_mut().rev() {
        t.value = r;
        r *= -1.0;
    }

    ts
}
