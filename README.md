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
- [ ] save checkpoints
- [x] eval checkpoints
- [ ] dirichlet noise for exploration
- [x] move RNG initialization outside of mcts constructor
- [x] sample move during training
- [x] cache evals during rollout
- [ ] sliding windows of games instead of full regeneration
- [x] compare NN against optimal dataset
- [ ] calculate c_puct instead of hardcoding
- [ ] recalculate best child while backproping