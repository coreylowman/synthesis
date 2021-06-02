pub trait HasTurnOrder: Eq + Clone + Copy + std::fmt::Debug {
    fn prev(&self) -> Self;
    fn next(&self) -> Self;
}

pub trait Env<const N: usize>: Clone {
    type PlayerId: HasTurnOrder;
    type Action: Eq + Clone + Copy + std::fmt::Debug + Into<usize> + From<usize>;
    type ActionIterator: Iterator<Item = Self::Action>;
    type State: Eq + std::hash::Hash + Clone;

    const MAX_NUM_ACTIONS: usize = N;
    const NAME: &'static str;
    const NUM_PLAYERS: usize;

    fn new() -> Self;
    fn get_state_dims() -> Vec<i64>;

    fn player(&self) -> Self::PlayerId;
    fn is_over(&self) -> bool;
    fn reward(&self, player_id: Self::PlayerId) -> f32;
    fn iter_actions(&self) -> Self::ActionIterator;
    fn step(&mut self, action: &Self::Action) -> bool;
    fn state(&self) -> Self::State;
    fn print(&self);
}
