use crate::env::Env;
use crate::mcts::{Policy, MCTS};
use rand::{distributions::Distribution, distributions::WeightedIndex, Rng};

#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    pub capacity: usize,
    pub num_explores: usize,
    pub temperature: f32,
    pub sample_action: bool,
    pub steps_per_epoch: usize,
}

pub fn run_game<E: Env + Clone, P: Policy<E>, R: Rng>(
    cfg: &RunConfig,
    policy: &P,
    rng: &mut R,
    states: &mut Vec<f32>,
    pis: &mut Vec<f32>,
    vs: &mut Vec<f32>,
) {
    let start_i = vs.len();
    let mut mcts = MCTS::<E, P>::with_capacity(cfg.capacity, policy);
    let mut game = E::new();
    let start_player = game.player();
    let mut is_over = false;
    let mut policy = vec![0.0; E::MAX_NUM_ACTIONS];

    while !is_over {
        let dur = mcts.explore_n(cfg.num_explores);
        // println!("{:?}", dur);

        let root_node = mcts.root_node();

        // save timestep
        let mut total = 0.0;
        policy.fill(0.0);
        for &(action, child_id) in root_node.children.iter() {
            let child = mcts.get_node(child_id);
            let value = child.num_visits.powf(1.0 / cfg.temperature);
            policy[action.into()] = value;
            total += value;
        }
        for i in 0..E::MAX_NUM_ACTIONS {
            policy[i] /= total;
        }

        states.extend(&root_node.state);
        pis.extend(&policy);
        vs.push(0.0);

        let action = if cfg.sample_action {
            let dist = WeightedIndex::new(&policy).unwrap();
            let choice = dist.sample(rng);
            E::Action::from(choice)
        } else {
            mcts.best_action()
        };
        mcts.step_action(&action);

        // println!("-----");
        // println!("Applying action {:?}", action);
        is_over = game.step(&action);
        // game.print();
    }

    let mut r = game.reward(start_player);
    for i in start_i..vs.len() {
        vs[i] = r;
        r *= -1.0;
    }
}

pub fn gather_experience<E: Env + Clone, P: Policy<E>, R: Rng>(
    cfg: &RunConfig,
    policy: &P,
    rng: &mut R,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut states: Vec<f32> = Vec::with_capacity(cfg.steps_per_epoch * 2);
    let mut pis: Vec<f32> = Vec::with_capacity(cfg.steps_per_epoch * 2);
    let mut vs: Vec<f32> = Vec::with_capacity(cfg.steps_per_epoch * 2);

    while vs.len() < cfg.steps_per_epoch {
        run_game(cfg, policy, rng, &mut states, &mut pis, &mut vs);
        // println!("{:?}", states.len());
    }

    // let states_t = Tensor::stack(&states, 0);
    // assert!(states_t.size()[0] == states.len() as i64);
    // assert!(states_t.size()[1..] == states[0].size());

    // let pis_t = Tensor::stack(&pis, 0);
    // assert!(pis_t.size()[0] == pis.len() as i64);
    // assert!(pis_t.size()[1..] == pis[0].size());

    // let vs_t = Tensor::of_slice(&vs).unsqueeze(1);
    // assert!(vs_t.size().len() == 2);
    // assert!(vs_t.size()[0] == vs.len() as i64);
    // assert!(vs_t.size()[1] == 1);

    // (states_t, pis_t, vs_t)
    (states, pis, vs)
}
