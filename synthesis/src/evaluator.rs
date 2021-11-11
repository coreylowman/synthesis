use crate::config::*;
use crate::game::*;
use crate::mcts::MCTS;
use crate::policies::*;
use crate::utils::*;
use rand::prelude::{Rng, SeedableRng, StdRng};
use tch::nn::VarStore;

pub fn evaluator<G: Game<N>, P: Policy<G, N> + NNPolicy<G, N>, const N: usize>(
    cfg: &EvaluationConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    std::thread::sleep(std::time::Duration::from_secs(1));

    let models_dir = cfg.logs.join("models");
    let pgn_path = cfg.logs.join("results.pgn");
    let mut pgn = std::fs::File::create(&pgn_path)?;
    let _guard = tch::no_grad_guard();
    let first_player = G::new().player();

    let mut best_k = Vec::with_capacity(cfg.num_best_policies);

    for i_iter in 0.. {
        // add new games for baselines so they don't fall behind
        {
            let i = i_iter % cfg.rollout_num_explores.len();
            for j in 0..cfg.rollout_num_explores.len() {
                if i == j {
                    continue;
                }
                let seed = i_iter;
                add_pgn_result(
                    &mut pgn,
                    &format!("VanillaMCTS{}", cfg.rollout_num_explores[i]),
                    &format!("VanillaMCTS{}", cfg.rollout_num_explores[j]),
                    mcts_vs_mcts::<G, N>(
                        &cfg,
                        first_player,
                        cfg.rollout_num_explores[i],
                        cfg.rollout_num_explores[j],
                        seed as u64,
                    ),
                )?;
            }
            calculate_ratings(&cfg.logs)?;
            plot_ratings(&cfg.logs)?;
        }

        // wait for model to exist;
        let name = format!("model_{}.ot", i_iter);
        while !models_dir.join(&name).exists() {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        // wait an extra second to be sure data is there
        std::thread::sleep(std::time::Duration::from_secs(1));

        // load model
        let mut vs = VarStore::new(tch::Device::Cpu);
        let mut policy = P::new(&vs);
        vs.load(models_dir.join(&name))?;

        // evaluate against rollout mcts
        for &explores in cfg.rollout_num_explores.iter() {
            let op_name = format!("VanillaMCTS{}", explores);
            for seed in 0..cfg.num_games_against_rollout {
                let result = eval_against_rollout_mcts(
                    &cfg,
                    &mut policy,
                    first_player,
                    explores,
                    seed as u64,
                );
                add_pgn_result(&mut pgn, &name, &op_name, result)?;
                let result = eval_against_rollout_mcts(
                    &cfg,
                    &mut policy,
                    first_player.next(),
                    explores,
                    seed as u64,
                );
                add_pgn_result(&mut pgn, &op_name, &name, result)?;
            }
            calculate_ratings(&cfg.logs)?;
            plot_ratings(&cfg.logs)?;
        }

        // evaluate against best old policies
        for (prev_name, prev_p) in best_k.iter_mut() {
            let result = eval_against_old(&cfg, &mut policy, prev_p);
            add_pgn_result(&mut pgn, &name, &prev_name, result)?;

            let result = eval_against_old(&cfg, prev_p, &mut policy);
            add_pgn_result(&mut pgn, &prev_name, &name, result)?;
        }

        // update results
        calculate_ratings(&cfg.logs)?;
        plot_ratings(&cfg.logs)?;

        // update top k
        if best_k.len() < cfg.num_best_policies {
            best_k.push((name, policy));
        } else {
            let ranks = rankings(&cfg.logs)?;
            if ranks
                .iter()
                .take(cfg.num_best_policies)
                .position(|n| n == &name)
                .is_some()
            {
                best_k.push((name, policy));
                match best_k.iter().position(|(n, _p)| {
                    ranks
                        .iter()
                        .take(cfg.num_best_policies)
                        .position(|n1| n1 == n)
                        .is_none()
                }) {
                    Some(i) => {
                        best_k.remove(i);
                    }
                    None => panic!("Didn't find policy to evict"),
                }
            }
        }
    }

    Ok(())
}

fn eval_against_old<G: Game<N>, P: Policy<G, N>, const N: usize>(
    cfg: &EvaluationConfig,
    p1: &mut P,
    p2: &mut P,
) -> f32 {
    let mut game = G::new();
    let first_player = game.player();
    loop {
        let action = if game.player() == first_player {
            MCTS::exploit(
                cfg.policy_num_explores,
                cfg.policy_mcts_cfg,
                p1,
                game.clone(),
                cfg.policy_action,
            )
        } else {
            MCTS::exploit(
                cfg.policy_num_explores,
                cfg.policy_mcts_cfg,
                p2,
                game.clone(),
                cfg.policy_action,
            )
        };
        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

fn eval_against_rollout_mcts<G: Game<N>, P: Policy<G, N>, const N: usize>(
    cfg: &EvaluationConfig,
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
                cfg.policy_num_explores,
                cfg.policy_mcts_cfg,
                policy,
                game.clone(),
                cfg.policy_action,
            )
        } else {
            FrozenMCTS::exploit(
                opponent_explores,
                cfg.rollout_mcts_cfg,
                &mut rollout_policy,
                game.clone(),
                cfg.rollout_action,
            )
        };

        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

fn mcts_vs_mcts<G: Game<N>, const N: usize>(
    cfg: &EvaluationConfig,
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
        let action = FrozenMCTS::exploit(
            if game.player() == player {
                p1_explores
            } else {
                p2_explores
            },
            cfg.rollout_mcts_cfg,
            &mut rollout_policy,
            game.clone(),
            cfg.rollout_action,
        );
        if game.step(&action) {
            break;
        }
    }
    game.reward(first_player)
}

type NodeId = u32;
type ActionId = u8;

#[derive(Debug)]
struct Node<G: Game<N>, const N: usize> {
    parent: NodeId,            // 4 bytes
    first_child: NodeId,       // 4 bytes
    num_children: u8,          // 1 byte
    game: G,                   // ? bytes
    solution: Option<Outcome>, // 1 byte
    action: ActionId,          // 1 byte
    action_prob: f32,          // 4 bytes
    cum_value: f32,            // 4 bytes
    num_visits: f32,           // 4 bytes
}

impl<G: Game<N>, const N: usize> Node<G, N> {
    fn unvisited(
        parent: NodeId,
        game: G,
        solution: Option<Outcome>,
        action: u8,
        action_prob: f32,
    ) -> Self {
        Self {
            parent,
            first_child: 0,
            num_children: 0,
            game,
            action,
            solution,
            action_prob,
            cum_value: 0.0,
            num_visits: 0.0,
        }
    }

    #[inline]
    fn is_unvisited(&self) -> bool {
        self.num_children == 0 && self.solution.is_none()
    }

    #[inline]
    fn is_visited(&self) -> bool {
        self.num_children != 0
    }

    #[inline]
    fn is_unsolved(&self) -> bool {
        self.solution.is_none()
    }

    #[inline]
    fn last_child(&self) -> NodeId {
        self.first_child + self.num_children as u32
    }

    #[inline]
    fn mark_visited(&mut self, first_child: NodeId, num_children: u8) {
        self.first_child = first_child;
        self.num_children = num_children;
    }

    #[inline]
    fn mark_solved(&mut self, outcome: Outcome) {
        self.solution = Some(outcome);
    }
}

pub struct FrozenMCTS<'a, G: Game<N>, P: Policy<G, N>, const N: usize> {
    root: NodeId,
    offset: NodeId,
    nodes: Vec<Node<G, N>>,
    policy: &'a mut P,
    cfg: MCTSConfig,
}

impl<'a, G: Game<N>, P: Policy<G, N>, const N: usize> FrozenMCTS<'a, G, P, N> {
    pub fn exploit(
        explores: usize,
        cfg: MCTSConfig,
        policy: &'a mut P,
        game: G,
        action_selection: ActionSelection,
    ) -> G::Action {
        let mut mcts = Self::with_capacity(explores + 1, cfg, policy, game);
        mcts.explore_n(explores);
        mcts.best_action(action_selection)
    }

    pub fn with_capacity(capacity: usize, cfg: MCTSConfig, policy: &'a mut P, game: G) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(Node::unvisited(0, game, None, 0, 0.0));
        let mut mcts = Self {
            root: 0,
            offset: 0,
            nodes,
            policy,
            cfg,
        };
        let (value, any_solved) = mcts.visit(mcts.root);
        mcts.backprop(mcts.root, value, any_solved);
        mcts
    }

    fn next_node_id(&self) -> NodeId {
        self.nodes.len() as NodeId + self.offset
    }

    fn node(&self, node_id: NodeId) -> &Node<G, N> {
        &self.nodes[(node_id - self.offset) as usize]
    }

    fn mut_node(&mut self, node_id: NodeId) -> &mut Node<G, N> {
        &mut self.nodes[(node_id - self.offset) as usize]
    }

    fn children_of(&self, node: &Node<G, N>) -> &[Node<G, N>] {
        &self.nodes
            [(node.first_child - self.offset) as usize..(node.last_child() - self.offset) as usize]
    }

    fn mut_nodes(&mut self, first_child: NodeId, last_child: NodeId) -> &mut [Node<G, N>] {
        &mut self.nodes[(first_child - self.offset) as usize..(last_child - self.offset) as usize]
    }

    pub fn best_action(&self, action_selection: ActionSelection) -> G::Action {
        let root = self.node(self.root);

        let mut best_action = None;
        let mut best_value = f32::NEG_INFINITY;
        for child in self.children_of(root) {
            if child.is_unvisited() {
                continue;
            }
            let value = match child.solution {
                Some(Outcome::Win(_)) => f32::NEG_INFINITY,
                Some(Outcome::Draw(_)) => 1e6,
                Some(Outcome::Lose(_)) => f32::INFINITY,
                None => match action_selection {
                    ActionSelection::Q => -child.cum_value / child.num_visits,
                    ActionSelection::NumVisits => child.num_visits,
                },
            };
            if best_action.is_none() || value > best_value {
                best_value = value;
                best_action = Some((child.action as usize).into());
            }
        }
        best_action.unwrap()
    }

    fn explore(&mut self) {
        let mut node_id = self.root;
        loop {
            let node = self.node(node_id);
            if let Some(outcome) = node.solution {
                self.backprop(node_id, outcome.value(), true);
                return;
            } else if node.is_unvisited() {
                let (value, any_solved) = self.visit(node_id);
                self.backprop(node_id, value, any_solved);
                return;
            } else {
                node_id = self.select_best_child(node_id);
            }
        }
    }

    fn select_best_child(&mut self, node_id: NodeId) -> NodeId {
        let node = self.node(node_id);

        let mut best_child_id = None;
        let mut best_value = f32::NEG_INFINITY;
        for child_ind in 0..node.num_children {
            let child_id = node.first_child + child_ind as u32;
            let child = self.node(child_id);
            let value = if child.is_unvisited() {
                let f = match self.cfg.fpu {
                    Fpu::Const(value) => value,
                    _ => panic!("Unsupported fpu in baseline"),
                };
                f + child.action_prob
            } else {
                let q = match child.solution {
                    Some(outcome) => outcome.reversed().value(),
                    None => -child.cum_value / child.num_visits,
                };
                let u = match self.cfg.exploration {
                    Exploration::Uct { c } => {
                        let visits = (c * node.num_visits.ln()).sqrt();
                        visits / child.num_visits.sqrt()
                    }
                    _ => {
                        panic!("Not supported in frozen mcts");
                    }
                };
                q + u
            };
            if best_child_id.is_none() || value > best_value {
                best_child_id = Some(child_id);
                best_value = value;
            }
        }
        best_child_id.unwrap()
    }

    fn visit(&mut self, node_id: NodeId) -> (f32, bool) {
        let first_child = self.next_node_id();
        let node = self.node(node_id);
        let game = node.game.clone();
        let (logits, dist) = self.policy.eval(&game);
        let mut num_children = 0;
        let mut any_solved = false;
        let mut max_logit = f32::NEG_INFINITY;
        for action in game.iter_actions() {
            let mut child_game = game.clone();
            let is_over = child_game.step(&action);
            let solution = if is_over {
                any_solved = true;
                Some(child_game.reward(child_game.player()).into())
            } else {
                None
            };
            let action: usize = action.into();
            let logit = logits[action];
            max_logit = max_logit.max(logit);
            let child = Node::unvisited(node_id, child_game, solution, action as u8, logit);
            self.nodes.push(child);
            num_children += 1;
        }

        let node = self.mut_node(node_id);
        node.mark_visited(first_child, num_children);
        let first_child = node.first_child;
        let last_child = node.last_child();

        // stable softmax
        let mut total = 0.0;
        for child in self.mut_nodes(first_child, last_child) {
            child.action_prob = (child.action_prob - max_logit).exp();
            total += child.action_prob;
        }
        for child in self.mut_nodes(first_child, last_child) {
            child.action_prob /= total;
        }

        let value = dist[2] - dist[0];

        (value, any_solved)
    }

    fn backprop(&mut self, leaf_node_id: NodeId, mut value: f32, mut solved: bool) {
        let mut node_id = leaf_node_id;
        loop {
            let node = self.node(node_id);
            let parent = node.parent;

            if self.cfg.solve && solved && node.is_unsolved() {
                let mut all_solved = true;
                let mut worst_solution = None;
                for child in self.children_of(node) {
                    if child.is_unvisited() || child.is_unsolved() {
                        all_solved = false;
                    } else if worst_solution.is_none() || child.solution < worst_solution {
                        worst_solution = child.solution;
                    }
                }

                let node = self.mut_node(node_id);
                if let Some(Outcome::Lose(_)) = worst_solution {
                    // at least 1 is a win, so mark this node as a win
                    node.mark_solved(Outcome::Win(0));
                    value = -node.cum_value + (node.num_visits + 1.0);
                } else if node.is_visited() && all_solved {
                    // all children node's are proven losses or draws
                    let best_for_me = worst_solution.unwrap().reversed();
                    node.mark_solved(best_for_me);
                    if let Outcome::Draw(_) = best_for_me {
                        value = -node.cum_value;
                    } else {
                        value = -node.cum_value - (node.num_visits + 1.0);
                    }
                } else {
                    solved = false;
                }
            }

            let node = self.mut_node(node_id);
            node.cum_value += value;
            node.num_visits += 1.0;
            value = -value;
            if node_id == self.root {
                break;
            }
            node_id = parent;
        }
    }

    pub fn explore_n(&mut self, n: usize) {
        for _ in 0..n {
            self.explore();
        }
    }
}
