use crate::config::{MCTSConfig, MCTSExploration};
use crate::game::{Game, Outcome};
use crate::policies::Policy;
use rand::{distributions::Distribution, Rng};
use rand_distr::Dirichlet;

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

pub struct MCTS<'a, G: Game<N>, P: Policy<G, N>, const N: usize> {
    root: NodeId,
    offset: NodeId,
    nodes: Vec<Node<G, N>>,
    policy: &'a mut P,
    cfg: MCTSConfig,
}

impl<'a, G: Game<N>, P: Policy<G, N>, const N: usize> MCTS<'a, G, P, N> {
    pub fn exploit(explores: usize, cfg: MCTSConfig, policy: &'a mut P, game: G) -> G::Action {
        let mut mcts = Self::with_capacity(explores + 1, cfg, policy, game);
        mcts.explore_n(explores);
        mcts.best_action()
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

    pub fn solution(&self, action: &G::Action) -> Option<Outcome> {
        let action: usize = (*action).into();
        let action = action as u8;
        let root = self.node(self.root);
        for child in self.children_of(root) {
            if child.is_unvisited() {
                continue;
            }
            if child.action == action {
                return child.solution;
            }
        }
        None
    }

    pub fn extract_search_policy(&self, search_policy: &mut [f32; N]) {
        let root = self.node(self.root);
        let mut total = 0.0;
        search_policy.fill(0.0);
        for child in self.children_of(root) {
            search_policy[child.action as usize] = child.num_visits;
            total += child.num_visits;
        }
        for i in 0..N {
            search_policy[i] /= total;
        }
    }

    pub fn extract_avg_value(&self) -> f32 {
        let root = self.node(self.root);
        match root.solution {
            Some(outcome) => outcome.value(),
            None => root.cum_value / root.num_visits,
        }
    }

    pub fn add_dirichlet_noise<R: Rng>(&mut self, rng: &mut R, alpha: f32, noise_weight: f32) {
        let (first_child, last_child, num_children) = {
            let root = self.node(self.root);
            (root.first_child, root.last_child(), root.num_children)
        };
        if num_children < 2 {
            return;
        }
        let dirichlet = Dirichlet::new_with_size(alpha, num_children as usize).unwrap();
        let noise_probs = dirichlet.sample(rng);
        for (noise, child) in noise_probs
            .iter()
            .zip(self.mut_nodes(first_child, last_child))
        {
            child.action_prob = child.action_prob * (1.0 - noise_weight) + noise_weight * noise;
        }
    }

    pub fn best_action(&self) -> G::Action {
        let root = self.node(self.root);

        let mut best_action = None;
        let mut best_value = f32::NEG_INFINITY;
        for child in self.children_of(root) {
            if child.is_unvisited() {
                continue;
            }
            let value = match child.solution {
                Some(Outcome::Win) => f32::NEG_INFINITY,
                Some(Outcome::Draw) => 1e6,
                Some(Outcome::Lose) => f32::INFINITY,
                None => -child.cum_value / child.num_visits,
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
                self.cfg.fpu + child.action_prob
            } else {
                let q = match child.solution {
                    Some(outcome) => outcome.reversed().value(),
                    None => -child.cum_value / child.num_visits,
                };
                let u = match self.cfg.exploration {
                    MCTSExploration::UCT { c } => {
                        let visits = (c * node.num_visits.ln()).sqrt();
                        visits / child.num_visits.sqrt()
                    }
                    MCTSExploration::PUCT { c } => {
                        let visits = node.num_visits.sqrt();
                        c * child.action_prob * visits / (1.0 + child.num_visits)
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
        let (logits, value) = self.policy.eval(&game);
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
                if worst_solution == Some(Outcome::Lose) {
                    // at least 1 is a win, so mark this node as a win
                    node.mark_solved(Outcome::Win);
                    value = -node.cum_value + (node.num_visits + 1.0);
                } else if node.is_visited() && all_solved {
                    // all children node's are proven losses or draws
                    let best_for_me = worst_solution.unwrap().reversed();
                    node.mark_solved(best_for_me);
                    if best_for_me == Outcome::Draw {
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

#[cfg(test)]
mod tests {
    use rand::prelude::{SeedableRng, StdRng};

    use super::*;
    use crate::game::HasTurnOrder;
    use crate::policies::RolloutPolicy;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, std::hash::Hash, PartialOrd, Ord)]
    pub enum PlayerId {
        X,
        O,
    }

    impl HasTurnOrder for PlayerId {
        fn prev(&self) -> Self {
            self.next()
        }

        fn next(&self) -> Self {
            match self {
                PlayerId::O => PlayerId::X,
                PlayerId::X => PlayerId::O,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    struct Action {
        row: usize,
        col: usize,
    }

    impl From<usize> for Action {
        fn from(i: usize) -> Self {
            let row = i / 3;
            let col = i % 3;
            Self { row, col }
        }
    }

    impl Into<usize> for Action {
        fn into(self) -> usize {
            self.row * 3 + self.col
        }
    }

    #[derive(Debug, PartialEq, Eq, std::hash::Hash, Clone)]
    struct TicTacToe {
        board: [[Option<PlayerId>; 3]; 3],
        player: PlayerId,
        turn: usize,
    }

    struct ActionIterator {
        game: TicTacToe,
        i: usize,
    }

    impl Iterator for ActionIterator {
        type Item = Action;

        fn next(&mut self) -> Option<Self::Item> {
            while self.i < 9 {
                let action: Action = self.i.into();
                self.i += 1;
                if self.game.board[action.row][action.col].is_none() {
                    return Some(action);
                }
            }

            None
        }
    }

    impl TicTacToe {
        fn won(&self, player: PlayerId) -> bool {
            let p = Some(player);
            if self.board[0][0] == p && self.board[0][1] == p && self.board[0][2] == p {
                return true;
            }
            if self.board[1][0] == p && self.board[1][1] == p && self.board[1][2] == p {
                return true;
            }
            if self.board[2][0] == p && self.board[2][1] == p && self.board[2][2] == p {
                return true;
            }
            if self.board[0][0] == p && self.board[1][0] == p && self.board[2][0] == p {
                return true;
            }
            if self.board[0][1] == p && self.board[1][1] == p && self.board[2][1] == p {
                return true;
            }
            if self.board[0][2] == p && self.board[1][2] == p && self.board[2][2] == p {
                return true;
            }
            if self.board[0][0] == p && self.board[1][1] == p && self.board[2][2] == p {
                return true;
            }
            if self.board[0][2] == p && self.board[1][1] == p && self.board[2][0] == p {
                return true;
            }

            false
        }
    }

    impl Game<9> for TicTacToe {
        type PlayerId = PlayerId;
        type Action = Action;
        type ActionIterator = ActionIterator;
        type Features = [[[f32; 3]; 3]; 3];

        const MAX_NUM_ACTIONS: usize = 9;
        const NAME: &'static str = "TicTacToe";
        const NUM_PLAYERS: usize = 2;
        const DIMS: &'static [i64] = &[3, 3, 3];

        fn new() -> Self {
            Self {
                board: [[None; 3]; 3],
                player: PlayerId::X,
                turn: 0,
            }
        }

        fn player(&self) -> Self::PlayerId {
            self.player
        }

        fn is_over(&self) -> bool {
            self.won(self.player) || self.won(self.player.prev()) || self.turn == 9
        }

        fn reward(&self, player_id: Self::PlayerId) -> f32 {
            if self.won(player_id) {
                1.0
            } else if self.won(player_id.next()) {
                -1.0
            } else {
                0.0
            }
        }

        fn iter_actions(&self) -> Self::ActionIterator {
            ActionIterator {
                game: self.clone(),
                i: 0,
            }
        }
        fn step(&mut self, action: &Self::Action) -> bool {
            assert!(action.row < 3);
            assert!(action.col < 3);
            assert!(self.board[action.row][action.col].is_none());
            self.board[action.row][action.col] = Some(self.player);
            self.player = self.player.next();
            self.turn += 1;
            self.is_over()
        }

        fn features(&self) -> Self::Features {
            let mut s = [[[0.0; 3]; 3]; 3];
            for row in 0..3 {
                for col in 0..3 {
                    if let Some(p) = self.board[row][col] {
                        if p == self.player {
                            s[0][row][col] = 1.0;
                        } else {
                            s[1][row][col] = 1.0;
                        }
                    } else {
                        s[2][row][col] = 1.0;
                    }
                }
            }
            s
        }

        fn print(&self) {
            for row in 0..3 {
                for col in 0..3 {
                    print!(
                        "{}",
                        match self.board[row][col] {
                            Some(PlayerId::X) => "x",
                            Some(PlayerId::O) => "o",
                            None => ".",
                        }
                    );
                }
                println!();
            }
            println!();
        }
    }

    // https://en.wikipedia.org/wiki/Tic-tac-toe

    #[test]
    fn test_solve_win() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut policy = RolloutPolicy { rng: &mut rng };
        let mut game = TicTacToe::new();
        game.step(&Action { row: 0, col: 0 });
        game.step(&Action { row: 0, col: 2 });
        let mut mcts = MCTS::with_capacity(
            1601,
            MCTSConfig {
                exploration: MCTSExploration::PUCT { c: 2.0 },
                solve: true,
                fpu: f32::INFINITY,
            },
            &mut policy,
            game.clone(),
        );
        while mcts.node(mcts.root).solution.is_none() {
            mcts.explore();
        }
        let mut search_policy = [0.0; 9];
        mcts.extract_search_policy(&mut search_policy);
        assert_eq!(mcts.node(mcts.root).solution, Some(Outcome::Win));
        assert_eq!(
            &search_policy,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(mcts.solution(&0.into()), None);
        assert_eq!(mcts.solution(&1.into()), None);
        assert_eq!(mcts.solution(&2.into()), None);
        assert_eq!(mcts.solution(&3.into()), None);
        assert_eq!(mcts.solution(&4.into()), None);
        assert_eq!(mcts.solution(&5.into()), None);
        assert_eq!(mcts.solution(&6.into()), Some(Outcome::Lose));
        assert_eq!(mcts.solution(&7.into()), None);
        assert_eq!(mcts.solution(&8.into()), None);
        assert_eq!(mcts.extract_avg_value(), 1.0);
        assert_eq!(mcts.best_action(), 6.into());
        assert_eq!(mcts.nodes.len(), 311);
    }

    #[test]
    fn test_solve_loss() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut policy = RolloutPolicy { rng: &mut rng };
        let mut game = TicTacToe::new();
        game.step(&Action { row: 0, col: 0 });
        game.step(&Action { row: 0, col: 2 });
        game.step(&Action { row: 2, col: 0 });
        let mut mcts = MCTS::with_capacity(
            1601,
            MCTSConfig {
                exploration: MCTSExploration::PUCT { c: 2.0 },
                solve: true,
                fpu: f32::INFINITY,
            },
            &mut policy,
            game.clone(),
        );
        while mcts.node(mcts.root).solution.is_none() {
            mcts.explore();
        }
        assert_eq!(mcts.node(mcts.root).solution, Some(Outcome::Lose));
        let mut search_policy = [0.0; 9];
        mcts.extract_search_policy(&mut search_policy);
        assert_eq!(
            &search_policy,
            &[
                0.0, 0.16666667, 0.0, 0.16666667, 0.16666667, 0.16666667, 0.0, 0.16666667,
                0.16666667
            ]
        );
        assert_eq!(mcts.solution(&0.into()), None);
        assert_eq!(mcts.solution(&1.into()), Some(Outcome::Win));
        assert_eq!(mcts.solution(&2.into()), None);
        assert_eq!(mcts.solution(&3.into()), Some(Outcome::Win));
        assert_eq!(mcts.solution(&4.into()), Some(Outcome::Win));
        assert_eq!(mcts.solution(&5.into()), Some(Outcome::Win));
        assert_eq!(mcts.solution(&6.into()), None);
        assert_eq!(mcts.solution(&7.into()), Some(Outcome::Win));
        assert_eq!(mcts.solution(&8.into()), Some(Outcome::Win));
        assert_eq!(mcts.extract_avg_value(), -1.0);
        assert_eq!(mcts.best_action(), 1.into());
        assert_eq!(mcts.nodes.len(), 69);
    }

    #[test]
    fn test_solve_draw() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut policy = RolloutPolicy { rng: &mut rng };
        let mut game = TicTacToe::new();
        game.step(&Action { row: 0, col: 0 });
        game.step(&Action { row: 1, col: 1 });
        let mut mcts = MCTS::with_capacity(
            1601,
            MCTSConfig {
                exploration: MCTSExploration::PUCT { c: 2.0 },
                solve: true,
                fpu: f32::INFINITY,
            },
            &mut policy,
            game.clone(),
        );
        while mcts.node(mcts.root).solution.is_none() {
            mcts.explore();
        }

        assert_eq!(mcts.node(mcts.root).solution, Some(Outcome::Draw));
        let mut search_policy = [0.0; 9];
        mcts.extract_search_policy(&mut search_policy);
        assert_eq!(
            &search_policy,
            &[
                0.0, 0.14285715, 0.14285715, 0.14285715, 0.0, 0.14285715, 0.14285715, 0.14285715,
                0.14285715
            ]
        );
        assert_eq!(mcts.solution(&0.into()), None);
        assert_eq!(mcts.solution(&1.into()), Some(Outcome::Draw));
        assert_eq!(mcts.solution(&2.into()), Some(Outcome::Draw));
        assert_eq!(mcts.solution(&3.into()), Some(Outcome::Draw));
        assert_eq!(mcts.solution(&4.into()), None);
        assert_eq!(mcts.solution(&5.into()), Some(Outcome::Draw));
        assert_eq!(mcts.solution(&6.into()), Some(Outcome::Draw));
        assert_eq!(mcts.solution(&7.into()), Some(Outcome::Draw));
        assert_eq!(mcts.solution(&8.into()), Some(Outcome::Draw));
        assert_eq!(mcts.extract_avg_value(), 0.0);
        assert_eq!(mcts.best_action(), 1.into());
        assert_eq!(mcts.nodes.len(), 1533);
    }

    #[test]
    fn test_add_noise() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut policy = RolloutPolicy { rng: &mut rng };
        let game = TicTacToe::new();
        let mut mcts = MCTS::with_capacity(
            1601,
            MCTSConfig {
                exploration: MCTSExploration::PUCT { c: 2.0 },
                solve: true,
                fpu: f32::INFINITY,
            },
            &mut policy,
            game.clone(),
        );
        let mut rng2 = StdRng::seed_from_u64(0);

        let mut total = 0.0;
        for child in mcts.children_of(mcts.node(mcts.root)) {
            assert!(child.action_prob > 0.0);
            total += child.action_prob;
        }
        assert!((total - 1.0).abs() < 1e-6);

        mcts.add_dirichlet_noise(&mut rng2, 1.0, 0.25);
        let mut total = 0.0;
        for child in mcts.children_of(mcts.node(mcts.root)) {
            assert!(child.action_prob > 0.0);
            total += child.action_prob;
        }
        assert!((total - 1.0).abs() < 1e-6);
    }
}
