Rust Implementation of AlphaZero
--------------------------------

https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
https://www.nature.com/articles/nature24270.epdf
https://dselsam.github.io/posts/2018-06-06-issues-with-alpha-zero.html
https://github.com/deepmind/open_spiel/blob/master/open_spiel/algorithms/alpha_zero_torch/alpha_zero.cc
https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/

This repo is a cargo workspace made up of multiple crates & binaries:

- ragz: The main crate with all the main training & MCTS logic in it
- riches: A binary that uses the ragz library & holds game specific code/NNs for each game
- base65536: A small crate to encode/decode u8's into valid utf-8 strings
- slimnn: A small neural network crate in pure rust
- invest: A binary that saves pytorch weights into a format slimnn can understand

TODOS

- [x] conv net in rust
- [x] PUCT exploration in mcts
- [x] use NN in mcts to assign probability to node
- [x] use visit count to select moves
- [x] Runner using nn params and mcts
- [x] main train loop of run games -> sample -> train
- [x] save checkpoints
- [x] eval checkpoints
- [x] dirichlet noise for exploration
- [x] move RNG initialization outside of mcts constructor
- [x] sample move during training
- [x] cache evals during rollout
- [x] sliding windows of games instead of full regeneration
- [x] compare NN against optimal dataset
- [x] rollout & eval separate processes
- [x] self play ELO
- [x] library for NN CPU execution & serialization/deserialization to string
- [x] improve NN specification & log NN structure
- [x] log game
- [x] fill buffer with random games at first
- [x] Use fraction of Q value + end game value.
  - [x] linear interpolate based on turn in game where 1st turn gets q value, last turn gets v
- [x] MCTS with solver
  - [ ] abort games if solver figures out if outcome is decided
- [ ] minimax value target (instead of Q)
- [ ] smart deduplication in training data (ie average all stats for each state)
- [ ] move order by action prob while expanding a node
- [ ] Assume un-evaluated node (FPU) as -1
- [ ] separate exploration/exploitation
  - [ ] exploration process that builds off of self play line by sampling other states backward
  - [ ] exploit process that samples a state from ^ and exploits all the way down
- [ ] change sample_actions_until throughout training
  - [ ] schedule
  - [ ] sample range
- [ ] change learning rate throughout training (cyclic? linear decrease?)
- [ ] calculate c_puct instead of hardcoding

Neural Network Architecture
- [ ] Value head as distribution of {W,D,L}
- [ ] output
  - [ ] clamp instead of tanh?
  - [ ] no final activation for value, but clamp in eval()?

Quality of life todos
- [ ] save replay buffer along with weights so you can resume training
- [x] make a general MCTS struct
- [ ] squash runner rollout functions into 1

Efficiency todos
- [ ] speed up conv2d with im2col https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
  - [ ] https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
- [ ] reverse linear weight dimensions for speed up
- [ ] C4 63x3 -> 256 -> 16 (first 9 policy, 10th value, last 6 unused for speed)
- [ ] transposition table hash
- [ ] support outputting 16 bit floats instead of 32 bit floats https://github.com/starkat99/half-rs/blob/master/src/bfloat/convert.rs

- [ ] look into https://github.com/leela-zero/leela-zero/issues/1156



```
queue of steps
for X number of games:
  pop step
  restore game to step's state
  play through game exploitatively, adding steps to queue
```