use crate::env::{Env, HasTurnOrder};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlayerId {
    Red,
    Black,
}

impl HasTurnOrder for PlayerId {
    fn prev(&self) -> Self {
        self.next()
    }

    fn next(&self) -> Self {
        match self {
            PlayerId::Black => PlayerId::Red,
            PlayerId::Red => PlayerId::Black,
        }
    }
}

const WIDTH: usize = 7;
const HEIGHT: usize = 6;

#[derive(Clone)]
pub struct Connect4 {
    my_bb: u64,
    op_bb: u64,
    height: [u8; WIDTH],
    player: PlayerId,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Column(u8);

impl From<usize> for Column {
    fn from(x: usize) -> Self {
        Column(x as u8)
    }
}

impl Into<usize> for Column {
    fn into(self) -> usize {
        self.0 as usize
    }
}
pub struct FreeColumns {
    height: [u8; WIDTH],
    col: u8,
}

impl Iterator for FreeColumns {
    type Item = Column;
    fn next(&mut self) -> Option<Self::Item> {
        if self.col == WIDTH as u8 {
            return None;
        }

        while self.col < WIDTH as u8 {
            if self.height[self.col as usize] < HEIGHT as u8 {
                let item = Some(Column(self.col));
                self.col += 1;
                return item;
            }
            self.col += 1;
        }

        None
    }
}

impl Connect4 {
    fn winner(&self) -> Option<PlayerId> {
        if self.won(self.my_bb) {
            Some(self.player)
        } else if self.won(self.op_bb) {
            Some(self.player.next())
        } else {
            None
        }
    }

    fn won(&self, bb: u64) -> bool {
        let d1 = bb & (bb >> 6) & (bb >> 12) & (bb >> 18);
        let d2 = bb & (bb >> 8) & (bb >> 16) & (bb >> 24);
        let h = bb & (bb >> 7) & (bb >> 14) & (bb >> 21);
        let v = bb & (bb >> 1) & (bb >> 2) & (bb >> 3);
        v + h + d1 + d2 > 0
    }
}

impl Env for Connect4 {
    const MAX_NUM_ACTIONS: usize = WIDTH;
    const NUM_PLAYERS: usize = 2;

    type PlayerId = PlayerId;
    type Action = Column;
    type ActionIterator = FreeColumns;

    fn new() -> Self {
        Self {
            my_bb: 0,
            op_bb: 0,
            height: [0; WIDTH],
            player: PlayerId::Red,
        }
    }

    fn player(&self) -> Self::PlayerId {
        self.player
    }

    fn is_over(&self) -> bool {
        self.winner().is_some() || (0..WIDTH).all(|col| self.height[col] == HEIGHT as u8)
    }

    fn reward(&self, player_id: Self::PlayerId) -> f32 {
        // assert!(self.is_over());

        match self.winner() {
            Some(winner) => {
                if winner == player_id {
                    1.0
                } else {
                    -1.0
                }
            }
            None => 0.0,
        }
    }

    fn iter_actions(&self) -> Self::ActionIterator {
        FreeColumns {
            height: self.height,
            col: 0,
        }
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        let col: usize = (*action).into();

        self.my_bb ^= 1 << (self.height[col] + 7 * (col as u8));
        self.height[col] += 1;

        std::mem::swap(&mut self.my_bb, &mut self.op_bb);
        self.player = self.player.next();

        self.is_over()
    }

    fn get_state_dims() -> Vec<i64> {
        vec![2, HEIGHT as i64, WIDTH as i64]
    }

    fn state(&self) -> Vec<f32> {
        let mut s = Vec::with_capacity(WIDTH * HEIGHT);
        for bb in &[self.my_bb, self.op_bb] {
            for row in 0..HEIGHT {
                for col in 0..WIDTH {
                    let index = 1 << (row + 7 * col);
                    if bb & index != 0 {
                        s.push(1.0);
                    } else {
                        s.push(0.0);
                    }
                }
            }
        }
        s
    }

    fn print(&self) {
        if self.is_over() {
            println!("{:?} won", self.winner());
        } else {
            println!("{:?} to play", self.player);
            println!(
                "Available Actions: {:?}",
                self.iter_actions().collect::<Vec<Column>>()
            );
        }

        let (my_char, op_char) = match self.player {
            PlayerId::Black => ("B", "r"),
            PlayerId::Red => ("r", "B"),
        };

        for row in (0..HEIGHT).rev() {
            for col in 0..WIDTH {
                let index = 1 << (row + 7 * col);
                print!(
                    "{}",
                    if self.my_bb & index != 0 {
                        my_char
                    } else if self.op_bb & index != 0 {
                        op_char
                    } else {
                        "."
                    }
                );
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connect4() {
        let mut game = Connect4::new();
        game.print();
        let s = game.state();
        assert!(s.iter().all(|&c| c == 0.0));
        assert!(game.player() == PlayerId::Red);

        assert!(game.step(&Column(0)) == false);
        game.print();
        let s = game.state();
        assert!(s[0] == -1.0);
        assert!(s[1..].iter().all(|&c| c == 0.0));

        assert!(game.step(&Column(2)) == false);
        game.print();
        let s = game.state();
        assert!(s[0] == 1.0);
        assert!(s[2] == -1.0);
    }
}
