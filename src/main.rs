mod connect4;
mod data;
mod env;
mod mcts;
mod model;
mod runner;
mod utils;

use crate::connect4::Connect4;
use crate::data::{tensor, BatchRandSampler};
use crate::env::Env;
use crate::mcts::Policy;
use crate::model::{ConvNet, NNPolicy};
use crate::runner::{eval, gather_experience, RolloutConfig};
use crate::utils::{
    add_pgn_result, calculate_ratings, git_diff, git_hash, save, save_str, train_dir,
};
use env_logger;
use log::{debug, info, trace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::default::Default;
use std::io::Write;
use tch::{
    kind::Kind,
    nn::{Adam, OptimizerConfig, VarStore},
};

#[derive(Debug, Serialize, Deserialize)]
struct TrainConfig {
    pub lr: f64,
    pub weight_decay: f64,
    pub num_iterations: usize,
    pub num_epochs: usize,
    pub batch_size: i64,
    pub seed: u64,
    pub logs: &'static str,
}

struct PolicyStorage {
    pub store: HashMap<String, VarStore>,
}

impl PolicyStorage {
    fn with_capacity(n: usize) -> Self {
        Self {
            store: HashMap::with_capacity(n),
        }
    }

    fn insert(&mut self, name: &String, vs: &VarStore) {
        let mut stored_vs = VarStore::new(tch::Device::Cpu);
        stored_vs.copy(vs).unwrap();
        self.store.insert(name.clone(), stored_vs);
    }

    fn get<E: Env, P: Policy<E> + NNPolicy<E>>(&self, name: &String) -> P {
        P::new(self.store.get(name).unwrap())
    }
}

fn train<E: Env, P: Policy<E> + NNPolicy<E>>(
    train_cfg: &TrainConfig,
    rollout_cfg: &RolloutConfig,
    solved: Option<Vec<tch::Tensor>>,
) {
    let train_dir = train_dir(train_cfg.logs);
    save(&train_dir, "train_cfg.json", train_cfg);
    save(&train_dir, "rollout_cfg.json", rollout_cfg);
    save_str(&train_dir, "git_hash", &git_hash());
    save_str(&train_dir, "git_diff.txt", &git_diff());
    let mut progress = std::fs::File::create(&train_dir.join("progress.csv")).expect("?");
    let pgn_path = train_dir.join("results.pgn");
    let mut pgn = std::fs::File::create(&pgn_path).expect("?");

    tch::manual_seed(train_cfg.seed as i64);
    let mut rng = StdRng::seed_from_u64(train_cfg.seed);

    let mut policies = PolicyStorage::with_capacity(train_cfg.num_iterations);

    let vs = VarStore::new(tch::Device::Cpu);
    let mut policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, train_cfg.lr).unwrap();
    opt.set_weight_decay(train_cfg.weight_decay);

    let mut state_dims = E::get_state_dims();
    state_dims.insert(0, -1);
    let mut action_dims = [-1, E::MAX_NUM_ACTIONS as i64];
    let mut value_dims = [-1, 1];

    info!("{:?}", rollout_cfg);
    info!("{:?}", train_cfg);

    let acc = {
        let _guard = tch::no_grad_guard();
        solved.as_ref().map_or(None, |ts| {
            let (_pis, vs) = policy.forward(&ts[0]);
            Some(f32::from(&(vs - &ts[1]).square().mean(Kind::Float)))
        })
    };
    info!("Init eval mse={:?}", acc);
    write!(progress, "0,{:?}\n", acc).expect("");

    let mut num_steps_trained = 0;
    let mut name = String::from("model_0.ot");
    policies.insert(&name, &vs);
    vs.save(train_dir.join(name)).unwrap();
    for i_iter in 0..train_cfg.num_iterations {
        debug!("Iteration {}...", i_iter);

        // gather data
        let (states, target_pis, target_vs) = {
            let _guard = tch::no_grad_guard();
            gather_experience::<E, P, StdRng>(rollout_cfg, &mut policy, &mut rng)
        };

        // convert to tensors
        state_dims[0] = target_vs.len() as i64;
        action_dims[0] = target_vs.len() as i64;
        value_dims[0] = target_vs.len() as i64;
        let states = tensor(&states, &state_dims, Kind::Float);
        let target_pis = tensor(&target_pis, &action_dims, Kind::Float);
        let target_vs = tensor(&target_vs, &value_dims, Kind::Float);

        for i_epoch in 0..train_cfg.num_epochs {
            let sampler =
                BatchRandSampler::new(&states, &target_pis, &target_vs, train_cfg.batch_size, true);

            // train
            let mut pi_eloss = 0f32;
            let mut v_eloss = 0f32;
            for (state, target_pi, target_v) in sampler {
                let (pi, v) = policy.forward(&state);

                let pi_loss = -(target_pi * pi.log()).sum(Kind::Float) / train_cfg.batch_size;
                let v_loss = (v - target_v).square().mean(Kind::Float);
                trace!("{:?} {:?}", pi_loss, v_loss);

                let loss = &pi_loss + &v_loss;
                opt.backward_step(&loss);

                pi_eloss += f32::from(&pi_loss);
                v_eloss += f32::from(&v_loss);
                num_steps_trained += train_cfg.batch_size;
            }
            debug!(
                "\tEpoch {} pi_loss={} v_loss={}",
                i_epoch, pi_eloss, v_eloss
            );
        }

        // evaluate against previous models
        {
            let _guard = tch::no_grad_guard();
            name = format!("model_{}.ot", i_iter + 1);
            for old_name in policies.store.keys() {
                let mut old_p = policies.get(old_name);
                let white_reward = eval(rollout_cfg, &mut policy, &mut old_p);
                let black_reward = eval(rollout_cfg, &mut old_p, &mut policy);
                add_pgn_result(&mut pgn, &name, old_name, white_reward);
                add_pgn_result(&mut pgn, old_name, &name, black_reward);
            }
            calculate_ratings(&train_dir).expect("Rating calculation failed");
        }

        policies.insert(&name, &vs);
        vs.save(train_dir.join(name)).unwrap();

        let acc = {
            let _guard = tch::no_grad_guard();
            solved.as_ref().map_or(None, |ts| {
                let (_pis, vs) = policy.forward(&ts[0]);
                Some(f32::from(&(vs - &ts[1]).square().mean(Kind::Float)))
            })
        };
        info!("Iteration {} eval mse={:?}", i_iter, acc);
        write!(progress, "{},{:?}\n", i_iter + 1, acc).expect("");
    }
}

fn main() {
    env_logger::init();

    let path = std::path::Path::new("./connect-4.npz");
    assert!(path.exists());
    let named_tensors = tch::Tensor::read_npz(path).unwrap();
    let solved = Some(
        named_tensors
            .iter()
            .map(|(_n, t)| t.to_kind(tch::Kind::Float))
            .collect(),
    );

    let train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 1e-5,
        num_iterations: 100,
        num_epochs: 16,
        batch_size: 256,
        seed: 0,
        logs: "./logs",
    };

    let rollout_cfg = RolloutConfig {
        capacity: 100_000,
        num_explores: 100,
        temperature: 1.0,
        sample_action: true,
        steps: 3200,
    };

    train::<Connect4, ConvNet<Connect4>>(&train_cfg, &rollout_cfg, solved);
}
