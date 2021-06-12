use crate::env::Env;
use crate::policies::Policy;
use std::time::{Duration, Instant};

pub struct Node<E: Env<N>, const N: usize> {
    pub parent: usize,
    pub env: E,
    pub terminal: bool,
    pub expanded: bool,
    pub actions: E::ActionIterator,
    pub children: Vec<(E::Action, usize)>,
    pub cum_value: f32,
    pub num_visits: f32,
    pub action_probs: [f32; N], // TODO make this a box?
    pub value: f32,
}

impl<E: Env<N>, const N: usize> Node<E, N> {
    pub fn new(
        parent_id: usize,
        env: E,
        is_over: bool,
        action_probs: [f32; N],
        value: f32,
    ) -> Self {
        let actions = env.iter_actions();
        let value = if is_over {
            env.reward(env.player())
        } else {
            value
        };
        Node {
            parent: parent_id,
            env,
            terminal: is_over,
            expanded: is_over,
            actions,
            children: Vec::new(),
            num_visits: 1.0,
            cum_value: value,
            action_probs,
            value,
        }
    }
}

pub struct MCTS<'a, E: Env<N>, P: Policy<E, N>, const N: usize> {
    pub root: usize,
    pub nodes: Vec<Node<E, N>>,
    pub states: Vec<E::State>,
    pub policy: &'a mut P,
    pub c_puct: f32,
}

impl<'a, E: Env<N>, P: Policy<E, N>, const N: usize> MCTS<'a, E, P, N> {
    pub fn with_capacity(capacity: usize, c_puct: f32, policy: &'a mut P, env: E) -> Self {
        let state = env.state();
        let (action_probs, value) = policy.eval(&state);
        let root = Node::new(0, env, false, action_probs, value);

        let mut nodes = Vec::with_capacity(capacity);
        let mut states = Vec::with_capacity(capacity);
        nodes.push(root);
        states.push(state);

        Self {
            root: 0,
            nodes,
            states,
            policy,
            c_puct,
        }
    }

    fn next_node_id(&self) -> usize {
        self.nodes.len() + self.root
    }

    pub fn add_noise(&mut self, noise: &Vec<f32>, noise_weight: f32) {
        let root = &mut self.nodes[self.root - self.root];
        for i in 0..N {
            root.action_probs[i] =
                (1.0 - noise_weight) * root.action_probs[i] + noise_weight * noise[i];
        }
    }

    pub fn step_action(&mut self, action: &E::Action) {
        // note: this function attempts to drop obviously unused nodes in order to reduce memory usage
        self.root = match self.nodes[self.root - self.root]
            .children
            .iter()
            .position(|(a, _)| a == action)
        {
            Some(action_index) => {
                let (_a, new_root) = self.nodes[self.root - self.root].children[action_index];
                drop(self.nodes.drain(0..new_root - self.root));
                drop(self.states.drain(0..new_root - self.root));
                new_root
            }
            None => {
                let mut env = self.nodes[self.root - self.root].env.clone();
                let is_over = env.step(action);
                let state = env.state();
                let (action_probs, value) = self.policy.eval(&state);
                let child_node = Node::new(0, env, is_over, action_probs, value);
                self.nodes.clear();
                self.states.clear();
                self.nodes.push(child_node);
                self.states.push(state);
                0
            }
        };

        self.nodes[0].parent = self.root;
    }

    pub fn root_node(&self) -> &Node<E, N> {
        self.get_node(self.root)
    }

    pub fn root_state(&self) -> &E::State {
        &self.states[self.root - self.root]
    }

    pub fn get_node(&self, node_id: usize) -> &Node<E, N> {
        &self.nodes[node_id - self.root]
    }

    pub fn best_action(&self) -> E::Action {
        let root = &self.nodes[self.root - self.root];

        let mut best_action = None;
        let mut best_value = -std::f32::INFINITY;

        for &(action, child_id) in root.children.iter() {
            let child = &self.nodes[child_id - self.root];
            let value = child.num_visits;
            if value > best_value {
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
            let node = &mut self.nodes[node_id - self.root];
            if node.terminal {
                let v = node.value;
                self.backprop(node_id, v);
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
                        let state = env.state();
                        let (action_probs, value) = self.policy.eval(&state);
                        let child = Node::new(node_id, env, is_over, action_probs, value);
                        self.nodes.push(child);
                        self.states.push(state);
                        self.backprop(node_id, -value);
                        return;
                    }
                    None => {
                        node.expanded = true;

                        // renormalize probabilities based on valid actions
                        let mut total = 0.0;
                        let mut mask = [0.0; N];
                        for &(action, _child_id) in &node.children {
                            mask[action.into()] = 1.0;
                            total += node.action_probs[action.into()];
                        }
                        for i in 0..N {
                            node.action_probs[i] *= mask[i] / total;
                        }

                        node_id = self.select_best_child(node_id);
                    }
                }
            }
        }
    }

    fn select_best_child(&mut self, node_id: usize) -> usize {
        let node = &self.nodes[node_id - self.root];

        let visits = node.num_visits.sqrt();

        let mut best_child_id = self.root;
        let mut best_value = -std::f32::INFINITY;
        for &(action, child_id) in &node.children {
            let child = &self.nodes[child_id - self.root];
            // NOTE: -child.cum_value because child.cum_value is in opponent's win pct, so we want to convert to ours
            let value = -child.cum_value / child.num_visits
                + self.c_puct * node.action_probs[action.into()] * visits
                    / (1.0 + child.num_visits);
            if value > best_value {
                best_child_id = child_id;
                best_value = value;
            }
        }
        best_child_id
    }

    fn backprop(&mut self, leaf_node_id: usize, mut value: f32) {
        let mut node_id = leaf_node_id;
        loop {
            let node = &mut self.nodes[node_id - self.root];

            node.num_visits += 1.0;
            node.cum_value += value;
            value *= -1.0;

            if node_id == self.root {
                break;
            }
            node_id = node.parent;
        }
    }

    pub fn explore_n(&mut self, n: usize) -> Duration {
        let start = Instant::now();
        for _ in 0..n {
            self.explore();
        }
        start.elapsed()
    }
}
