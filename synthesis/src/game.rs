use std::cmp::Ordering;
use std::hash::Hash;

pub trait HasTurnOrder: Eq + Clone + Copy + std::fmt::Debug {
    fn prev(&self) -> Self;
    fn next(&self) -> Self;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Outcome {
    Win(usize),
    Lose(usize),
    Draw(usize),
}

impl From<f32> for Outcome {
    fn from(value: f32) -> Self {
        if value > 0.0 {
            Self::Win(0)
        } else if value < 0.0 {
            Self::Lose(0)
        } else {
            Self::Draw(0)
        }
    }
}

impl Outcome {
    pub fn reversed(&self) -> Self {
        match self {
            Self::Win(u) => Self::Lose(*u + 1),
            Self::Lose(u) => Self::Win(*u + 1),
            Self::Draw(u) => Self::Draw(*u + 1),
        }
    }

    pub fn value(&self) -> f32 {
        match self {
            Self::Win(_) => 1.0,
            Self::Draw(_) => 0.0,
            Self::Lose(_) => -1.0,
        }
    }
}

impl Ord for Outcome {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Win(a), Self::Win(b)) => b.cmp(a), // NOTE: reversed, want to win in least number of terms
            (Self::Win(_a), Self::Draw(_b)) => Ordering::Greater,
            (Self::Win(_a), Self::Lose(_b)) => Ordering::Greater,
            (Self::Draw(_a), Self::Win(_b)) => Ordering::Less,
            (Self::Draw(a), Self::Draw(b)) => a.cmp(b),
            (Self::Draw(_a), Self::Lose(_b)) => Ordering::Greater,
            (Self::Lose(_a), Self::Win(_b)) => Ordering::Less,
            (Self::Lose(_a), Self::Draw(_b)) => Ordering::Less,
            (Self::Lose(a), Self::Lose(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for Outcome {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub trait Game<const N: usize>: Eq + Hash + Clone + std::fmt::Debug + Send {
    type PlayerId: HasTurnOrder;
    type Action: Eq + Clone + Copy + std::fmt::Debug + Into<usize> + From<usize>;
    type ActionIterator: Iterator<Item = Self::Action>;
    type Features: PartialEq + Clone + std::fmt::Debug + Send;

    const MAX_NUM_ACTIONS: usize = N;
    const MAX_TURNS: usize;
    const NAME: &'static str;
    const NUM_PLAYERS: usize;
    const DIMS: &'static [i64];

    fn new() -> Self;
    fn player(&self) -> Self::PlayerId;
    fn is_over(&self) -> bool;
    fn reward(&self, player_id: Self::PlayerId) -> f32;
    fn iter_actions(&self) -> Self::ActionIterator;
    fn step(&mut self, action: &Self::Action) -> bool;
    fn features(&self) -> Self::Features;
    fn print(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmp_outcome() {
        assert_eq!(Outcome::Win(0).cmp(&Outcome::Win(0)), Ordering::Equal);
        assert_eq!(Outcome::Win(0).cmp(&Outcome::Draw(0)), Ordering::Greater);
        assert_eq!(Outcome::Win(0).cmp(&Outcome::Lose(0)), Ordering::Greater);

        assert_eq!(Outcome::Draw(0).cmp(&Outcome::Win(0)), Ordering::Less);
        assert_eq!(Outcome::Draw(0).cmp(&Outcome::Draw(0)), Ordering::Equal);
        assert_eq!(Outcome::Draw(0).cmp(&Outcome::Lose(0)), Ordering::Greater);

        assert_eq!(Outcome::Lose(0).cmp(&Outcome::Win(0)), Ordering::Less);
        assert_eq!(Outcome::Lose(0).cmp(&Outcome::Draw(0)), Ordering::Less);
        assert_eq!(Outcome::Lose(0).cmp(&Outcome::Lose(0)), Ordering::Equal);
    }

    #[test]
    fn test_ord_outcome() {
        assert!(Outcome::Win(0) == Outcome::Win(0));
        assert!(Outcome::Win(0) > Outcome::Draw(0));
        assert!(Outcome::Win(0) > Outcome::Lose(0));

        assert!(Outcome::Draw(0) < Outcome::Win(0));
        assert!(Outcome::Draw(0) == Outcome::Draw(0));
        assert!(Outcome::Draw(0) > Outcome::Lose(0));

        assert!(Outcome::Lose(0) < Outcome::Win(0));
        assert!(Outcome::Lose(0) < Outcome::Draw(0));
        assert!(Outcome::Lose(0) == Outcome::Lose(0));
    }

    #[test]
    fn test_partial_ord_outcome() {
        assert!(Some(Outcome::Win(0)) > None);
        assert!(Some(Outcome::Draw(0)) > None);
        assert!(Some(Outcome::Lose(0)) > None);

        assert!(Some(Outcome::Win(0)) == Some(Outcome::Win(0)));
        assert!(Some(Outcome::Win(0)) > Some(Outcome::Draw(0)));
        assert!(Some(Outcome::Win(0)) > Some(Outcome::Lose(0)));

        assert!(Some(Outcome::Draw(0)) < Some(Outcome::Win(0)));
        assert!(Some(Outcome::Draw(0)) == Some(Outcome::Draw(0)));
        assert!(Some(Outcome::Draw(0)) > Some(Outcome::Lose(0)));

        assert!(Some(Outcome::Lose(0)) < Some(Outcome::Win(0)));
        assert!(Some(Outcome::Lose(0)) < Some(Outcome::Draw(0)));
        assert!(Some(Outcome::Lose(0)) == Some(Outcome::Lose(0)));
    }
}
