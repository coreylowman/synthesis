mod data;
mod envs;
mod mcts;
mod policies;
mod policy_impls;
mod runner;
mod utils;

use crate::data::{tensor, BatchRandSampler};
use crate::envs::{Connect4, Env};
use crate::policies::*;
use crate::policy_impls::*;
use crate::runner::{eval, gather_experience, ReplayBuffer, RolloutConfig};
use crate::utils::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::default::Default;
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
    pub buffer_size: usize,
    pub seed: u64,
    pub logs: &'static str,
}

fn train<E: Env, P: Policy<E> + NNPolicy<E>>(train_cfg: &TrainConfig, rollout_cfg: &RolloutConfig) {
    let train_dir = train_dir(train_cfg.logs);
    let models_dir = train_dir.join("models");
    let pgn_path = train_dir.join("results.pgn");
    let mut pgn = std::fs::File::create(&pgn_path).expect("?");

    std::fs::create_dir(&models_dir).expect("");
    save(&train_dir, "train_cfg.json", train_cfg);
    save(&train_dir, "rollout_cfg.json", rollout_cfg);
    save_str(&train_dir, "env_name", &E::NAME.into());
    save_str(&train_dir, "git_hash", &git_hash());
    save_str(&train_dir, "git_diff.txt", &git_diff());

    tch::manual_seed(train_cfg.seed as i64);
    let mut rng = StdRng::seed_from_u64(train_cfg.seed);

    let mut policies = PolicyStorage::with_capacity(train_cfg.num_iterations);
    let mut buffer = ReplayBuffer::<E>::new(train_cfg.buffer_size);

    let vs = VarStore::new(tch::Device::Cpu);
    let mut policy = P::new(&vs);
    let mut opt = Adam::default().build(&vs, train_cfg.lr).unwrap();
    opt.set_weight_decay(train_cfg.weight_decay);

    let mut dims = E::get_state_dims();
    let num_acs = E::MAX_NUM_ACTIONS as i64;

    let mut name = String::from("model_0.ot");
    policies.insert(&name, &vs);
    vs.save(models_dir.join(name)).unwrap();
    for i_iter in 0..train_cfg.num_iterations {
        // gather data
        {
            let _guard = tch::no_grad_guard();
            gather_experience::<E, P, StdRng>(rollout_cfg, &mut policy, &mut rng, &mut buffer);
        }

        // convert to tensors
        dims[0] = buffer.vs.len() as i64;
        let states = tensor(&buffer.states, &dims, Kind::Float);
        let target_pis = tensor(&buffer.pis, &[dims[0], num_acs], Kind::Float);
        let target_vs = tensor(&buffer.vs, &[dims[0], 1], Kind::Float);

        // train
        for _i_epoch in 0..train_cfg.num_epochs {
            let sampler =
                BatchRandSampler::new(&states, &target_pis, &target_vs, train_cfg.batch_size, true);

            let mut epoch_loss = [0.0, 0.0];
            for (state, target_pi, target_v) in sampler {
                let legal_mask = target_pi.greater1(&tch::Tensor::zeros_like(&target_pi));
                let illegal_value = -1e10f32 * tch::Tensor::ones_like(&target_pi);

                let (logits, v) = policy.forward(&state);

                let legal_logits = logits.where1(&legal_mask, &illegal_value);
                let log_pi = legal_logits.log_softmax(-1, Kind::Float);

                // let pi_loss = -(target_pi * log_pi).sum(Kind::Float) / train_cfg.batch_size;
                let pi_loss = -(target_pi * log_pi).mean(Kind::Float);
                let v_loss = (v - target_v).square().mean(Kind::Float);

                let loss = &pi_loss + &v_loss;
                opt.backward_step(&loss);

                epoch_loss[0] += f32::from(&pi_loss);
                epoch_loss[1] += f32::from(&v_loss);
            }
            // println!("{} {:?}", _i_epoch, epoch_loss);
        }

        // evaluate against previous models
        {
            let _guard = tch::no_grad_guard();
            name = format!("model_{}.ot", i_iter + 1);
            for old_name in policies.last(50) {
                let mut old_p = policies.get(old_name);
                let white_reward = eval(rollout_cfg, &mut policy, &mut old_p);
                let black_reward = eval(rollout_cfg, &mut old_p, &mut policy);
                add_pgn_result(&mut pgn, &name, old_name, white_reward);
                add_pgn_result(&mut pgn, old_name, &name, black_reward);
            }
            calculate_ratings(&train_dir).expect("Rating calculation failed");
        }

        policies.insert(&name, &vs);
        vs.save(models_dir.join(name)).unwrap();
    }
}

fn main() {
    let train_cfg = TrainConfig {
        lr: 1e-3,
        weight_decay: 1e-5,
        num_iterations: 100,
        num_epochs: 16,
        batch_size: 256,
        buffer_size: 16_000,
        seed: 0,
        logs: "./logs",
    };

    let rollout_cfg = RolloutConfig {
        capacity: 100_000,
        num_explores: 100,
        temperature: 1.0,
        sample_action: true,
        steps: 3_200,
        alpha: 1.0,
        noisy_explore: false,
        c_puct: 4.0,
    };

    train::<Connect4, Connect4Net>(&train_cfg, &rollout_cfg);
}
