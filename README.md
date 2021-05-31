Rust Implementation of AlphaZero
--------------------------------

https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go

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
- [ ] library for NN CPU execution & serialization/deserialization to string
- [ ] improve NN specification & log NN structure
- [x] log game
- [ ] improve state for UTTT
- [ ] improve state/action for breakthrough
- [ ] play against trained model in CLI