# Synthesis: A Rust implementation of AlphaZero

This repo is a cargo workspace made up of multiple crates & binaries:

- `synthesis`: The main library crate with all the main training & MCTS logic in it
- `study-connect4`: A binary crate that uses the synthesis library to train a network to play Connect4
- `base65536`: A small crate to encode/decode u8's into valid utf-8 strings
- `slimnn`: A small neural network crate in pure rust
- `export`: A binary crate that saves pytorch weights into a format slimnn can understand

```cargo run --release --bin study-connect4```

## What's implemented

- Integration with the tch-rs [1] package to support pytorch in rust
- 💪 General MCTS implementation that supports the standard rollout method as well as using a NN in place of rollouts
  - Includes MCTS Solver [2]
  - Includes FPU [3]
- 💡 An AlphaZero [4] learner that collects experience using MCTS+NN and trains a policy and value function
  - Supports multiple value targets
  - All hyperparameters exposed
  - Multi threaded support! 👩‍👩‍👧‍👧
- 📈 Lightweight evaluation against standard rollout mcts with various number of explores
  - Saves game outcomes to a pgn file
  - Runs bayeselo [5] executable to produce elo ratings
  - Plots ratings 🎉
- 🎲 9x7 Connect4 as a playground to test things
- 😎 Support for running without torch
  - `slimnn` for simple NN layer implementations
  - `export` & `base65536` for converting torch weights to utf-8 strings

1. https://github.com/LaurentMazare/tch-rs
2. Winands, Mark HM, Yngvi Björnsson, and Jahn-Takeshi Saito. "Monte-Carlo tree search solver." International Conference on Computers and Games. Springer, Berlin, Heidelberg, 2008.
3. Gelly, Sylvain, and Yizao Wang. "Exploration exploitation in go: UCT for Monte-Carlo go." NIPS: Neural Information Processing Systems Conference On-line trading of Exploration and Exploitation Workshop. 2006.
4. https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
5. https://www.remi-coulom.fr/Bayesian-Elo/

## Improvements

###### General

- [ ] Evaluation metrics in addition to elo:
  - [ ] Depth reached
  - [ ] Something for how quickly positions are solved
  - [ ] Search policy accuracy
  - [ ] value accuracy against Q
  - [ ] value accuracy against 2-ply minimax value
- [ ] mix mcst tree and minimax tree (of solved nodes) using p(correct)
- [ ] Support transpositions (and backprop to multiple parents) while training... does this improve strength?
- [ ] Score Bounded solver https://www.lamsade.dauphine.fr/~cazenave/papers/mcsolver.pdf
- [ ] Ordinal MCTS https://arxiv.org/pdf/1901.04274.pdf
- [ ] Regularized Policy Optimization https://arxiv.org/abs/2007.12509
- [ ] Schedules for various parameters
  - [ ] sample_actions_until
  - [ ] value target
  - [ ] noise_weight
- [ ] New algorithm for separate exploration/exploitation
  - [ ] Is this ExIt? https://arxiv.org/pdf/1705.08439.pdf
  - [ ] exploration process that builds off of exploit play line by sampling other states backward
  - [ ] exploit process that samples a state from ^ and exploits all the way down

###### Performance
- [x] compiler flags (LTO=fat, codegen-units=1, target=native)
- [x] multi threaded gather_experience
- [ ] Reduce allocations (pre allocated buffer for MCTS nodes?)
- [ ] speed up conv2d with im2col https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
  - [ ] https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
- [ ] reverse linear weight dimensions for speed up
- [ ] support outputting 16 bit floats instead of 32 bit floats https://github.com/starkat99/half-rs/blob/master/src/bfloat/convert.rs

## Resources for learning more about AlphaZero

- https://medium.com/@sleepsonthefloor/azfour-a-connect-four-webapp-powered-by-the-alphazero-algorithm-d0c82d6f3ae9
- https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
- https://www.nature.com/articles/nature24270.epdf
- https://dselsam.github.io/posts/2018-06-06-issues-with-alpha-zero.html
- https://github.com/deepmind/open_spiel/blob/master/open_spiel/algorithms/alpha_zero_torch/alpha_zero.cc
- https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/
- http://proceedings.mlr.press/v97/tian19a/tian19a.pdf
- https://link.springer.com/content/pdf/10.1007/s00521-021-05928-5.pdf

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
