use crate::config::*;
use crate::game::*;
use crate::mcts::MCTS;
use crate::policies::*;
use crate::utils::*;
use rand::prelude::{Rng, SeedableRng, StdRng};
use tch::nn::VarStore;

pub fn evaluator<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: &LearningConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let models_dir = cfg.logs.join("models");
    let pgn_path = cfg.logs.join("results.pgn");
    let mut pgn = std::fs::File::create(&pgn_path)?;
    let _guard = tch::no_grad_guard();
    let first_player = G::new().player();
    let all_explores = [
        100, 200, 400, 800, 1600, 2400, 3200, 4800, 6400, 9600, 12800,
    ];

    for i in 0..all_explores.len() {
        for j in 0..all_explores.len() {
            if i == j {
                continue;
            }
            for seed in 0..50 {
                add_pgn_result(
                    &mut pgn,
                    &format!("VanillaMCTS{}", all_explores[i]),
                    &format!("VanillaMCTS{}", all_explores[j]),
                    mcts_vs_mcts::<G, N>(
                        &cfg,
                        first_player,
                        all_explores[i],
                        all_explores[j],
                        seed,
                    ),
                )?;
            }
        }
    }

    for i_iter in 0..cfg.num_iterations + 1 {
        // wait for model to exist;
        let name = format!("model_{}.ot", i_iter);
        while !models_dir.join(&name).exists() {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        // wait an extra second to be sure data is there
        std::thread::sleep(std::time::Duration::from_secs(1));

        // load model
        let mut vs = VarStore::new(tch::Device::Cpu);
        let policy = P::new(&vs);
        vs.load(models_dir.join(&name))?;
        let mut policy = OwnedPolicyWithCache::with_capacity(100_000, policy);

        let result = eval_against_random(&cfg, &mut policy, first_player);
        add_pgn_result(&mut pgn, &name, &String::from("Random"), result)?;
        let result = eval_against_random(&cfg, &mut policy, first_player.next());
        add_pgn_result(&mut pgn, &String::from("Random"), &name, result)?;

        for &explores in &all_explores {
            let op_name = format!("VanillaMCTS{}", explores);
            for seed in 0..10 {
                let result =
                    eval_against_vanilla_mcts(&cfg, &mut policy, first_player, explores, seed);
                add_pgn_result(&mut pgn, &name, &op_name, result)?;
                let result = eval_against_vanilla_mcts(
                    &cfg,
                    &mut policy,
                    first_player.next(),
                    explores,
                    seed,
                );
                add_pgn_result(&mut pgn, &op_name, &name, result)?;
            }
        }

        // update results
        calculate_ratings(&cfg.logs)?;
        plot_ratings(&cfg.logs)?;
    }

    Ok(())
}

fn eval_against_random<G: Game<N>, P: Policy<G, N>, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    player: G::PlayerId,
) -> f32 {
    let mut game = G::new();
    let first_player = game.player();
    let mut opponent = StdRng::seed_from_u64(0);
    loop {
        let action = if game.player() == player {
            MCTS::exploit(
                cfg.num_explores,
                cfg.c_puct,
                cfg.solve,
                cfg.fpu,
                policy,
                game.clone(),
            )
        } else {
            let num_actions = game.iter_actions().count() as u8;
            let i = opponent.gen_range(0..num_actions) as usize;
            game.iter_actions().nth(i).unwrap()
        };

        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

fn eval_against_vanilla_mcts<G: Game<N>, P: Policy<G, N>, const N: usize>(
    cfg: &LearningConfig,
    policy: &mut P,
    player: G::PlayerId,
    opponent_explores: usize,
    seed: u64,
) -> f32 {
    let mut game = G::new();
    let first_player = game.player();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rollout_policy = RolloutPolicy { rng: &mut rng };
    loop {
        let action = if game.player() == player {
            MCTS::exploit(
                cfg.num_explores,
                cfg.c_puct,
                cfg.solve,
                cfg.fpu,
                policy,
                game.clone(),
            )
        } else {
            MCTS::exploit(
                opponent_explores,
                cfg.c_puct,
                cfg.solve,
                cfg.fpu,
                &mut rollout_policy,
                game.clone(),
            )
        };

        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

fn mcts_vs_mcts<G: Game<N>, const N: usize>(
    cfg: &LearningConfig,
    player: G::PlayerId,
    p1_explores: usize,
    p2_explores: usize,
    seed: u64,
) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rollout_policy = RolloutPolicy { rng: &mut rng };
    let mut game = G::new();
    let first_player = game.player();
    loop {
        let action = MCTS::exploit(
            if game.player() == player {
                p1_explores
            } else {
                p2_explores
            },
            cfg.c_puct,
            cfg.solve,
            cfg.fpu,
            &mut rollout_policy,
            game.clone(),
        );
        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}
