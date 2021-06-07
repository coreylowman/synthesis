import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    print(args.path)

    scores = {}
    with open(args.path) as fp:
        # skip header
        fp.readline()

        for line in fp:
            while "  " in line:
                line = line.replace("  ", " ", 1)
            parts = line.strip().split(" ")
            num = int(parts[1].split("_")[1].split(".")[0])
            elo = int(parts[2])
            scores[num] = elo

    names = sorted(scores)
    elos = [scores[name] for name in names]
    plt.plot(names, elos)
    plt.savefig(f"{os.path.dirname(args.path)}/ratings.png")


if __name__ == "__main__":
    main()