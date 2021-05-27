import numpy as np

VALUES = {
    "win": 1.0,
    "draw": 0.0,
    "loss": -1.0,
}


def make_obs(board):
    obs = []
    cols = [board[i * 6 : i * 6 + 6] for i in range(7)]
    for p in ["x", "o"]:
        for row in range(6):
            for col in range(7):
                cell = cols[col][row]
                if cell == p:
                    obs.append(1.0)
                else:
                    obs.append(0.0)
    return np.array(obs).reshape((2, 6, 7))


def main():
    states = []
    values = []
    with open("connect-4.data") as fp:
        for line in fp:
            *board, result = line.strip().split(",")
            value = VALUES[result]
            states.append(make_obs(board))
            values.append(value)

    states = np.stack(states)
    values = np.expand_dims(np.stack(values), -1)

    print(states.shape, values.shape)

    np.savez(
        "connect-4.npz", states=states.astype(np.float), values=values.astype(np.float)
    )


if __name__ == "__main__":
    main()
