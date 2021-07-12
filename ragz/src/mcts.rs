use crate::env::{Env, Outcome};
use crate::policies::Policy;

type NodeId = u32;

struct Node<E: Env<N>, const N: usize> {
    parent: NodeId,
    env: E,
    terminal: bool,
    expanded: bool,
    outcome: Option<Outcome>,
    action_probs: [f32; N],
    value: f32,
    actions: E::ActionIterator,
    children: Vec<(E::Action, NodeId)>,
    cum_value: f32,
    num_visits: f32,
}

impl<E: Env<N>, const N: usize> Node<E, N> {
    fn new(
        parent: NodeId,
        env: E,
        is_over: bool,
        outcome: Option<Outcome>,
        action_probs: [f32; N],
        value: f32,
    ) -> Self {
        let actions = env.iter_actions();
        Node {
            parent,
            env,
            terminal: is_over,
            expanded: is_over,
            outcome,
            action_probs,
            value,
            actions,
            children: Vec::new(),
            cum_value: value,
            num_visits: 1.0,
        }
    }
}

pub struct MCTS<'a, E: Env<N>, P: Policy<E, N>, const N: usize> {
    root: NodeId,
    offset: NodeId,
    nodes: Vec<Node<E, N>>,
    policy: &'a mut P,
    c_puct: f32,
    solve: bool,
}

impl<'a, E: Env<N>, P: Policy<E, N>, const N: usize> MCTS<'a, E, P, N> {
    pub fn exploit(
        explores: usize,
        c_puct: f32,
        solve: bool,
        policy: &'a mut P,
        env: E,
    ) -> E::Action {
        let mut mcts = Self::with_capacity(explores + 1, c_puct, solve, policy, env);
        mcts.explore_n(explores);
        mcts.best_action()
    }

    pub fn with_capacity(
        capacity: usize,
        c_puct: f32,
        solve: bool,
        policy: &'a mut P,
        env: E,
    ) -> Self {
        let (action_probs, value) = policy.eval(&env);
        let root = Node::new(0, env, false, None, action_probs, value);

        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(root);

        Self {
            root: 0,
            offset: 0,
            nodes,
            policy,
            c_puct,
            solve,
        }
    }

    fn next_node_id(&self) -> NodeId {
        self.nodes.len() as NodeId + self.offset
    }

    fn node(&self, node_id: NodeId) -> &Node<E, N> {
        &self.nodes[(node_id - self.offset) as usize]
    }

    fn mut_node(&mut self, node_id: NodeId) -> &mut Node<E, N> {
        &mut self.nodes[(node_id - self.offset) as usize]
    }

    pub fn outcome(&self, action: &E::Action) -> Option<Outcome> {
        let root = self.node(self.root);
        let child_ind = root
            .children
            .iter()
            .position(|(a, _child_id)| a == action)
            .unwrap();
        let child = self.node(root.children[child_ind].1);
        child.outcome
    }

    pub fn extract_search_policy(&self, search_policy: &mut [f32; N]) {
        search_policy.fill(0.0);
        let root = self.node(self.root);
        let mut total = 0.0;
        let mut any_solved = false;
        let mut any_won = false;
        let mut any_draw = false;
        let mut all_lost = true;
        for &(_action, child_id) in root.children.iter() {
            let child = self.node(child_id);
            match child.outcome {
                Some(outcome) => {
                    any_solved = true;
                    match outcome {
                        Outcome::Win => {}
                        Outcome::Draw => {
                            all_lost = false;
                            any_draw = true;
                        }
                        Outcome::Lose => {
                            all_lost = false;
                            any_won = true;
                        }
                    }
                }
                None => {
                    all_lost = false;
                }
            }
        }
        if any_solved && any_won {
            for &(action, child_id) in root.children.iter() {
                let child = self.node(child_id);
                let value = match child.outcome {
                    Some(Outcome::Lose) => 1.0,
                    _ => 0.0,
                };
                search_policy[action.into()] = value;
                total += value;
            }
        } else if any_solved && any_draw {
            for &(action, child_id) in root.children.iter() {
                let child = self.node(child_id);
                let value = match child.outcome {
                    Some(Outcome::Draw) => 1.0,
                    _ => 0.0,
                };
                search_policy[action.into()] = value;
                total += value;
            }
        } else if any_solved && all_lost {
            for &(action, _child_id) in root.children.iter() {
                search_policy[action.into()] = 1.0;
                total += 1.0;
            }
        } else if any_solved && !any_won && !any_draw {
            // some lost, some unsolved
            for &(action, child_id) in root.children.iter() {
                let child = self.node(child_id);
                let value = match child.outcome {
                    Some(Outcome::Win) => 0.0,
                    Some(Outcome::Draw) => panic!(),
                    Some(Outcome::Lose) => panic!(),
                    _ => child.num_visits,
                };
                search_policy[action.into()] = value;
                total += value;
            }
        } else {
            for &(action, child_id) in root.children.iter() {
                let child = self.node(child_id);
                search_policy[action.into()] = child.num_visits;
                total += child.num_visits;
            }
        }
        for i in 0..N {
            search_policy[i] /= total;
        }
    }

    pub fn extract_q(&self) -> f32 {
        let root = self.node(self.root);
        root.cum_value / root.num_visits
    }

    pub fn add_noise(&mut self, noise: &Vec<f32>, noise_weight: f32) {
        let root = self.mut_node(self.root);
        for i in 0..N {
            root.action_probs[i] =
                (1.0 - noise_weight) * root.action_probs[i] + noise_weight * noise[i];
        }
    }

    pub fn best_action(&self) -> E::Action {
        let root = self.node(self.root);

        let mut best_action = None;
        let mut best_value = f32::NEG_INFINITY;
        for &(action, child_id) in root.children.iter() {
            let child = self.node(child_id);
            let value = match child.outcome {
                Some(Outcome::Win) => f32::NEG_INFINITY,
                Some(Outcome::Draw) => 1e6,
                Some(Outcome::Lose) => f32::INFINITY,
                None => -child.cum_value / child.num_visits,
            };
            if best_action.is_none() || value > best_value {
                best_value = value;
                best_action = Some(action);
            }
        }
        best_action.unwrap()
    }

    fn explore(&mut self) {
        let child_id = self.next_node_id();
        let mut node_id = self.root;
        loop {
            let node = self.mut_node(node_id);
            if node.terminal {
                let v = node.value;
                self.backprop(node_id, v, true);
                return;
            } else if let Some(outcome) = node.outcome {
                self.backprop(node_id, outcome.value(), true);
                return;
            } else if node.expanded {
                node_id = self.select_best_child(node_id);
            } else {
                match node.actions.next() {
                    Some(action) => {
                        // add to children
                        node.children.push((action, child_id));

                        // create the child node... note we will be modifying num_visits and reward later, so mutable
                        let mut env = node.env.clone();
                        let is_over = env.step(&action);
                        let (logits, value, outcome) = if is_over {
                            let r = env.reward(env.player());
                            ([0.0; N], r, Some(r.into()))
                        } else {
                            let (logits, value) = self.policy.eval(&env);
                            (logits, value, None)
                        };
                        let child = Node::new(node_id, env, is_over, outcome, logits, value);
                        self.nodes.push(child);
                        self.backprop(node_id, -value, is_over);
                        return;
                    }
                    None => {
                        node.expanded = true;
                        node_id = self.select_best_child(node_id);
                    }
                }
            }
        }
    }

    fn select_best_child(&mut self, node_id: NodeId) -> NodeId {
        let node = self.node(node_id);

        let visits = node.num_visits.sqrt();

        let mut best_child_id = None;
        let mut best_value = f32::NEG_INFINITY;
        for &(action, child_id) in node.children.iter() {
            let child = self.node(child_id);
            // NOTE: -child.cum_value because child.cum_value is in opponent's win pct, so we want to convert to ours
            let value = match child.outcome {
                Some(outcome) => outcome.reversed().value(),
                None => {
                    let q = -child.cum_value / child.num_visits;
                    let u = node.action_probs[action.into()] * visits / (1.0 + child.num_visits);
                    q + self.c_puct * u
                }
            };
            if best_child_id.is_none() || value > best_value {
                best_child_id = Some(child_id);
                best_value = value;
            }
        }
        best_child_id.unwrap()
    }

    fn backprop(&mut self, leaf_node_id: NodeId, mut value: f32, mut solved: bool) {
        let mut node_id = leaf_node_id;
        loop {
            let node = self.node(node_id);
            let parent = node.parent;

            if self.solve && solved && node.outcome.is_none() {
                let mut all_solved = true;
                let mut worst_child_outcome = None;
                for &(_action, child_id) in node.children.iter() {
                    let child = self.node(child_id);
                    if child.outcome.is_none() {
                        all_solved = false;
                    } else if worst_child_outcome.is_none() || child.outcome < worst_child_outcome {
                        worst_child_outcome = child.outcome;
                    }
                }

                let node = self.mut_node(node_id);
                if worst_child_outcome == Some(Outcome::Lose) {
                    // at least 1 is a win, so mark this node as a win
                    node.outcome = Some(Outcome::Win);
                    value = -node.cum_value + (node.num_visits + 1.0);
                } else if node.expanded && all_solved {
                    // all children node's are proven losses or draws
                    let best_for_me = worst_child_outcome.unwrap().reversed();
                    node.outcome = Some(best_for_me);
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
    use crate::env::HasTurnOrder;
    use crate::policies::RolloutPolicy;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

    #[derive(Debug, PartialEq, Eq, Clone)]
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

    impl Env<9> for TicTacToe {
        type PlayerId = PlayerId;
        type Action = Action;
        type ActionIterator = ActionIterator;
        type State = [[[bool; 3]; 3]; 3];

        const MAX_NUM_ACTIONS: usize = 9;
        const NAME: &'static str = "TicTacToe";
        const NUM_PLAYERS: usize = 2;

        fn new() -> Self {
            Self {
                board: [[None; 3]; 3],
                player: PlayerId::X,
                turn: 0,
            }
        }

        fn restore(state: &Self::State) -> Self {
            Self::new()
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

        fn get_state_dims() -> Vec<i64> {
            vec![3, 3, 3]
        }

        fn state(&self) -> Self::State {
            let mut s = [[[false; 3]; 3]; 3];
            for row in 0..3 {
                for col in 0..3 {
                    if let Some(p) = self.board[row][col] {
                        if p == self.player {
                            s[0][row][col] = true;
                        } else {
                            s[1][row][col] = true;
                        }
                    } else {
                        s[2][row][col] = true;
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
        let mut mcts = MCTS::with_capacity(1601, 2.0, true, &mut policy, game.clone());
        while mcts.node(mcts.root).outcome.is_none() {
            mcts.explore();
        }
        let mut search_policy = [0.0; 9];
        mcts.extract_search_policy(&mut search_policy);
        assert_eq!(mcts.node(mcts.root).outcome, Some(Outcome::Win));
        assert_eq!(
            &search_policy,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(mcts.extract_q(), 1.0);
    }

    #[test]
    fn test_solve_loss() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut policy = RolloutPolicy { rng: &mut rng };
        let mut game = TicTacToe::new();
        game.step(&Action { row: 0, col: 0 });
        game.step(&Action { row: 0, col: 2 });
        game.step(&Action { row: 2, col: 0 });
        let mut mcts = MCTS::with_capacity(1601, 2.0, true, &mut policy, game.clone());
        while mcts.node(mcts.root).outcome.is_none() {
            mcts.explore();
        }
        assert_eq!(mcts.node(mcts.root).outcome, Some(Outcome::Lose));
        let mut search_policy = [0.0; 9];
        mcts.extract_search_policy(&mut search_policy);
        assert_eq!(
            &search_policy,
            &[
                0.0, 0.16666667, 0.0, 0.16666667, 0.16666667, 0.16666667, 0.0, 0.16666667,
                0.16666667
            ]
        );
        assert_eq!(mcts.extract_q(), -1.0);
    }

    #[test]
    fn test_solve_draw() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut policy = RolloutPolicy { rng: &mut rng };
        let mut game = TicTacToe::new();
        game.step(&Action { row: 0, col: 0 });
        game.step(&Action { row: 1, col: 1 });
        let mut mcts = MCTS::with_capacity(1601, 2.0, true, &mut policy, game.clone());
        while mcts.node(mcts.root).outcome.is_none() {
            mcts.explore();
        }

        assert_eq!(mcts.node(mcts.root).outcome, Some(Outcome::Draw));
        let mut search_policy = [0.0; 9];
        mcts.extract_search_policy(&mut search_policy);
        assert_eq!(
            &search_policy,
            &[
                0.0, 0.14285715, 0.14285715, 0.14285715, 0.0, 0.14285715, 0.14285715, 0.14285715,
                0.14285715
            ]
        );
        assert_eq!(mcts.extract_q(), 0.0);
    }
}
