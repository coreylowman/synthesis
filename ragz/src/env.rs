use std::cmp::Ordering;

pub trait HasTurnOrder: Eq + Clone + Copy + std::fmt::Debug {
    fn prev(&self) -> Self;
    fn next(&self) -> Self;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Outcome {
    Win,
    Lose,
    Draw,
}

impl From<f32> for Outcome {
    fn from(value: f32) -> Self {
        if value > 0.0 {
            Self::Win
        } else if value < 0.0 {
            Self::Lose
        } else {
            Self::Draw
        }
    }
}

impl Outcome {
    pub fn reversed(&self) -> Self {
        match self {
            Self::Win => Self::Lose,
            Self::Lose => Self::Win,
            Self::Draw => Self::Draw,
        }
    }

    pub fn value(&self) -> f32 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.0,
            Self::Lose => -1.0,
        }
    }
}

impl Ord for Outcome {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Win, Self::Win) => Ordering::Equal,
            (Self::Win, Self::Draw) => Ordering::Greater,
            (Self::Win, Self::Lose) => Ordering::Greater,
            (Self::Draw, Self::Win) => Ordering::Less,
            (Self::Draw, Self::Draw) => Ordering::Equal,
            (Self::Draw, Self::Lose) => Ordering::Greater,
            (Self::Lose, Self::Win) => Ordering::Less,
            (Self::Lose, Self::Draw) => Ordering::Less,
            (Self::Lose, Self::Lose) => Ordering::Equal,
        }
    }
}

impl PartialOrd for Outcome {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub trait Env<const N: usize>: Clone + std::fmt::Debug {
    type PlayerId: HasTurnOrder;
    type Action: Eq + Clone + Copy + std::fmt::Debug + Into<usize> + From<usize>;
    type ActionIterator: Iterator<Item = Self::Action>;
    type State: Eq + std::hash::Hash + Clone;

    const MAX_NUM_ACTIONS: usize = N;
    const NAME: &'static str;
    const NUM_PLAYERS: usize;

    fn new() -> Self;
    fn restore(state: &Self::State) -> Self;
    fn get_state_dims() -> Vec<i64>;

    fn player(&self) -> Self::PlayerId;
    fn is_over(&self) -> bool;
    fn reward(&self, player_id: Self::PlayerId) -> f32;
    fn iter_actions(&self) -> Self::ActionIterator;
    fn step(&mut self, action: &Self::Action) -> bool;
    fn state(&self) -> Self::State;
    fn print(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmp_outcome() {
        assert_eq!(Outcome::Win.cmp(&Outcome::Win), Ordering::Equal);
        assert_eq!(Outcome::Win.cmp(&Outcome::Draw), Ordering::Greater);
        assert_eq!(Outcome::Win.cmp(&Outcome::Lose), Ordering::Greater);

        assert_eq!(Outcome::Draw.cmp(&Outcome::Win), Ordering::Less);
        assert_eq!(Outcome::Draw.cmp(&Outcome::Draw), Ordering::Equal);
        assert_eq!(Outcome::Draw.cmp(&Outcome::Lose), Ordering::Greater);

        assert_eq!(Outcome::Lose.cmp(&Outcome::Win), Ordering::Less);
        assert_eq!(Outcome::Lose.cmp(&Outcome::Draw), Ordering::Less);
        assert_eq!(Outcome::Lose.cmp(&Outcome::Lose), Ordering::Equal);
    }

    #[test]
    fn test_ord_outcome() {
        assert!(Outcome::Win == Outcome::Win);
        assert!(Outcome::Win > Outcome::Draw);
        assert!(Outcome::Win > Outcome::Lose);

        assert!(Outcome::Draw < Outcome::Win);
        assert!(Outcome::Draw == Outcome::Draw);
        assert!(Outcome::Draw > Outcome::Lose);

        assert!(Outcome::Lose < Outcome::Win);
        assert!(Outcome::Lose < Outcome::Draw);
        assert!(Outcome::Lose == Outcome::Lose);
    }

    #[test]
    fn test_partial_ord_outcome() {
        assert!(Some(Outcome::Win) > None);
        assert!(Some(Outcome::Draw) > None);
        assert!(Some(Outcome::Lose) > None);

        assert!(Some(Outcome::Win) == Some(Outcome::Win));
        assert!(Some(Outcome::Win) > Some(Outcome::Draw));
        assert!(Some(Outcome::Win) > Some(Outcome::Lose));

        assert!(Some(Outcome::Draw) < Some(Outcome::Win));
        assert!(Some(Outcome::Draw) == Some(Outcome::Draw));
        assert!(Some(Outcome::Draw) > Some(Outcome::Lose));

        assert!(Some(Outcome::Lose) < Some(Outcome::Win));
        assert!(Some(Outcome::Lose) < Some(Outcome::Draw));
        assert!(Some(Outcome::Lose) == Some(Outcome::Lose));
    }
}
