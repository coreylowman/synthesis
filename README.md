Rust Implementation of AlphaZero
--------------------------------

https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go

https://www.nature.com/articles/nature24270.epdf

This repo is a cargo workspace made up of multiple crates & binaries:

- base65536: A small crate to encode/decode u8's into valid utf-8 strings
- slimnn: A small neural network crate in pure rust
- ragz: The main crate with all the main training logic in it
- riches: A binary that uses the ragz library
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
- [ ] recalculate best child while backproping
- [ ] multiple rollout processes
- [ ] multiple threads for MCTS
- [ ] rollout & eval separate processes
- [x] self play ELO
- [x] library for NN CPU execution & serialization/deserialization to string
- [x] improve NN specification & log NN structure
- [x] log game
- [x] improve state for UTTT
- [ ] improve state/action for breakthrough
- [ ] play against trained model in CLI
- [ ] MAE instead of MSE? 1 epoch instead of 2?