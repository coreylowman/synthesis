use crate::env::Env;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use tch::{IndexOp, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlayerId {
    Red,
    Black,
}

impl PlayerId {
    fn next(&self) -> PlayerId {
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
    board: [[Option<PlayerId>; HEIGHT]; WIDTH],
    player: PlayerId,
}

pub struct FreeColumns {
    game: Connect4,
    col: usize,
}

impl Iterator for FreeColumns {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.col == WIDTH {
            return None;
        }

        while self.col < WIDTH {
            if self.game.has_space(self.col) {
                let item = Some(self.col);
                self.col += 1;
                return item;
            }
            self.col += 1;
        }

        None
    }
}

struct HorzCoords {
    col: usize,
    row: usize,
    curr: usize,
}
impl Iterator for HorzCoords {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == 4 || self.col + self.curr >= WIDTH {
            return None;
        }
        let item = Some((self.row, self.col + self.curr));
        self.curr += 1;
        return item;
    }
}

struct VertCoords {
    col: usize,
    row: usize,
    curr: usize,
}
impl Iterator for VertCoords {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == 4 || self.row + self.curr >= HEIGHT {
            return None;
        }
        let item = Some((self.row + self.curr, self.col));
        self.curr += 1;
        return item;
    }
}

struct DiagRightCoords {
    col: usize,
    row: usize,
    curr: usize,
}
impl Iterator for DiagRightCoords {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == 4 || self.col + self.curr >= WIDTH || self.row + self.curr >= HEIGHT {
            return None;
        }
        let item = Some((self.row + self.curr, self.col + self.curr));
        self.curr += 1;
        return item;
    }
}

struct DiagLeftCoords {
    col: usize,
    row: usize,
    curr: usize,
}
impl Iterator for DiagLeftCoords {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == 4 || self.curr > self.col || self.row + self.curr >= HEIGHT {
            return None;
        }
        let item = Some((self.row + self.curr, self.col - self.curr));
        self.curr += 1;
        return item;
    }
}

impl Connect4 {
    fn at(&self, row: usize, col: usize) -> Option<PlayerId> {
        self.board[col][row]
    }

    fn has_space(&self, col: usize) -> bool {
        self.at(HEIGHT - 1, col).is_none()
    }

    fn drop(&mut self, col: usize) {
        for i in 0..HEIGHT {
            if self.board[col][i].is_none() {
                self.board[col][i] = Some(self.player);
                break;
            }
        }
    }

    fn winner(&self) -> Option<PlayerId> {
        if self.won(PlayerId::Black) {
            Some(PlayerId::Black)
        } else if self.won(PlayerId::Red) {
            Some(PlayerId::Red)
        } else {
            None
        }
    }

    fn won(&self, player: PlayerId) -> bool {
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                if self.won_with(player, HorzCoords { col, row, curr: 0 }) {
                    return true;
                } else if self.won_with(player, VertCoords { col, row, curr: 0 }) {
                    return true;
                } else if self.won_with(player, DiagLeftCoords { col, row, curr: 0 }) {
                    return true;
                } else if self.won_with(player, DiagRightCoords { col, row, curr: 0 }) {
                    return true;
                }
            }
        }
        false
    }

    fn won_with(&self, player: PlayerId, iter: impl Iterator<Item = (usize, usize)>) -> bool {
        let mut count = 0;
        for (row, col) in iter {
            if let Some(owner) = self.at(row, col) {
                if owner == player {
                    count += 1;
                    if count == 4 {
                        return true;
                    }
                } else {
                    count = 0;
                }
            }
        }
        false
    }
}

impl Env for Connect4 {
    const MAX_NUM_ACTIONS: usize = WIDTH;
    const NUM_PLAYERS: usize = 2;

    type PlayerId = PlayerId;
    type Action = usize;
    type ActionIterator = FreeColumns;

    fn new() -> Self {
        Self {
            board: [[None; HEIGHT]; WIDTH],
            player: PlayerId::Red,
        }
    }

    fn player(&self) -> Self::PlayerId {
        self.player
    }

    fn is_over(&self) -> bool {
        self.winner().is_some() || (0..WIDTH).all(|col| !self.has_space(col))
    }

    fn reward(&self, player_id: Self::PlayerId) -> f32 {
        assert!(self.is_over());

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
            game: self.clone(),
            col: 0,
        }
    }

    fn num_actions(&self) -> u8 {
        self.iter_actions().collect::<Vec<Self::Action>>().len() as u8
    }

    fn get_random_action(&self, rng: &mut StdRng) -> Self::Action {
        let actions = self.iter_actions().collect::<Vec<Self::Action>>();
        *actions.choose(rng).unwrap()
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        let col = *action;
        assert!(self.has_space(col));
        self.drop(col);
        self.player = self.player.next();
        self.is_over()
    }

    fn state(&self, kind: tch::Kind, device: tch::Device) -> Tensor {
        let mut t = Tensor::zeros(&[2, HEIGHT as i64, WIDTH as i64], (kind, device));
        let mut p = self.player();
        for i in 0..2i64 {
            for col in 0..WIDTH {
                for row in 0..HEIGHT {
                    if let Some(owner) = self.at(row, col) {
                        if owner == p {
                            t.i((i, row as i64, col as i64)).fill_(1.0);
                        }
                    }
                }
            }
            p = p.next();
        }
        t
    }

    fn print(&self) {
        if self.is_over() {
            println!("{:?} won", self.winner());
        } else {
            println!("{:?} to play", self.player);
            println!(
                "Available Actions: {:?}",
                self.iter_actions().collect::<Vec<usize>>()
            );
        }

        for row in (0..HEIGHT).rev() {
            for col in 0..WIDTH {
                print!(
                    "{} ",
                    match self.at(row, col) {
                        Some(PlayerId::Black) => "B",
                        Some(PlayerId::Red) => "r",
                        None => ".",
                    }
                );
            }
            println!();
        }
    }
}
