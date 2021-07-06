use crate::env::Env;
use crate::policies::Policy;

type NodeId = u32;

struct Node<E: Env<N>, const N: usize> {
    parent: NodeId,
    env: E,
    terminal: bool,
    expanded: bool,
    outcome: Option<bool>,
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
        outcome: Option<bool>,
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

    pub fn outcome(&self, action: &E::Action) -> Option<bool> {
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
        let root = self.node(self.root);
        search_policy.fill(0.0);
        let mut total = 0.0;
        for &(action, child_id) in root.children.iter() {
            let child = self.node(child_id);
            search_policy[action.into()] = child.num_visits;
            total += child.num_visits;
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
                Some(true) => f32::NEG_INFINITY,
                Some(false) => f32::INFINITY,
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
            } else if let Some(won) = node.outcome {
                let v = if won { 1.0 } else { -1.0 };
                self.backprop(node_id, v, true);
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
                            ([0.0; N], r, Some(r > 0.0))
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

                        // normalize probabilities based on valid actions
                        let mut total = 0.0;
                        let mut mask = [0.0; N];
                        for &(action, _child_id) in &node.children {
                            mask[action.into()] = 1.0;
                            total += node.action_probs[action.into()].exp();
                        }
                        for i in 0..N {
                            node.action_probs[i] = node.action_probs[i].exp() * mask[i] / total;
                        }

                        node_id = self.select_best_child(node_id);
                    }
                }
            }
        }
    }

    fn select_best_child(&mut self, node_id: NodeId) -> NodeId {
        let node = self.node(node_id);

        let visits = node.num_visits.sqrt();

        let mut best_child_id = self.root;
        let mut best_value = f32::NEG_INFINITY;
        for &(action, child_id) in node.children.iter() {
            let child = self.node(child_id);
            // NOTE: -child.cum_value because child.cum_value is in opponent's win pct, so we want to convert to ours
            let value = match child.outcome {
                Some(true) => 0.0,
                Some(false) => 1.0,
                None => {
                    0.5 + -0.5 * child.cum_value / child.num_visits
                        + self.c_puct * node.action_probs[action.into()] * visits
                            / (1.0 + child.num_visits)
                }
            };
            if value > best_value {
                best_child_id = child_id;
                best_value = value;
            }
        }
        best_child_id
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
                    } else if worst_child_outcome.is_none() || child.outcome == Some(false) {
                        worst_child_outcome = child.outcome;
                    }
                }

                let node = self.mut_node(node_id);
                if worst_child_outcome == Some(false) {
                    // at least 1 is a win, so mark this node as a win
                    node.outcome = Some(true);
                    value = -node.cum_value + (node.num_visits + 1.0);
                } else if node.expanded && all_solved && worst_child_outcome == Some(true) {
                    // all children node's are proven won
                    node.outcome = Some(false);
                    value = -node.cum_value - (node.num_visits + 1.0);
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
