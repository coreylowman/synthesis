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
- [ ] calculate c_puct instead of hardcoding
- [ ] multiple rollout processes
- [x] rollout & eval separate processes
- [x] self play ELO
- [x] library for NN CPU execution & serialization/deserialization to string
- [x] improve NN specification & log NN structure
- [x] log game
- [ ] MCTS with solver
- [ ] Assume un-evaluated node (FPU) as -1
- [ ] Use fraction of Q value + end game value
- [ ] Value head as distribution of {W,D,L}
- [ ] shuffle action iterator to reduce bias of first moves in action iterator
- [ ] fill buffer with random games at first

the optimistic excuse algorithm:
  - only give a reward of -1 if you are using all the best actions and still lose
  - if you use any explore actions, you auto get a reward of 1 for that game

the try hard and the explorer:
    - 1 player plays best moves, other player explores
    - explorer stores (v+q)/2, max(v, q), or q
    - try hard stores v

