import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    scores = {}
    random_score = None
    mcts_scores = {}
    with open(args.path) as fp:
        # skip header
        fp.readline()

        for line in fp:
            while "  " in line:
                line = line.replace("  ", " ", 1)
            parts = line.strip().split(" ")
            elo = int(parts[2])
            if parts[1] == "Random":
                random_score = elo
            elif "VanillaMCTS" in parts[1]:
                mcts_scores[parts[1]] = elo
            else:
                num = int(parts[1].split("_")[1].split(".")[0])
                scores[num] = elo

    names = sorted(scores)
    elos = [scores[name] - scores[0] for name in names]
    plt.plot(names, elos, label="Learner")
    plt.scatter(names, elos)
    for name, elo in mcts_scores.items():
        if elo - scores[0] < 0:
            continue
        plt.plot(
            [names[0], names[-1]],
            [elo - scores[0], elo - scores[0]],
            linestyle="dashed",
            label=name,
        )
        plt.text(names[-1], elo - scores[0], name.replace("VanillaMCTS", ""))
    # plt.legend()
    plt.title("Strength through training")
    plt.xlabel("Iteration")
    plt.ylim(bottom=-20)
    plt.ylabel("BayesianELO")
    plt.savefig(f"{os.path.dirname(args.path)}/ratings.png")


if __name__ == "__main__":
    main()
