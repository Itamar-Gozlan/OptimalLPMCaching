import matplotlib.pyplot as plt
import numpy as np

with_splice_05 = {"0.1": 79.11,
                  "0.25": 91.4,
                  "0.5": 91.82,
                  }

no_splice_05 = {"0.1": 79.18,
                "0.25": 91.2,
                "0.5": 91.56,
                }

with_splice_1 = {
    "0.1": 85.18,
    "0.25": 90.61,
    "0.5": 92.07
}

no_splice_1 = {"0.1": 86.38,
               "0.25": 90.55,
               "0.5": 91.9
               }


def plot_cache_hit():
    cache_size_array = ['0.1', '0.25', '0.5']
    marker = ["o", "s", "d", "X", "v", "^"]
    label = ['A', 'B', 'C']
    fig, ax = plt.subplots()

    ax.plot(with_splice_1.keys(), with_splice_1.values(), marker=marker[0], label="With Splice Epoch 1.0")
    ax.plot(no_splice_1.keys(), no_splice_1.values(), marker=marker[1], label="No Splice Epoch 1.0")
    ax.plot(with_splice_05.keys(), with_splice_05.values(), marker=marker[2], label="With Splice Epoch 0.5")
    ax.plot(no_splice_05.keys(), no_splice_05.values(), marker=marker[3], label="No Splice Epoch 0.5")

    ax.set_ylabel("Cache-hit ratio (%)")
    ax.set_xlabel("Cache Size (%)")

    # ax.set_title("Epoch 1.0")
    ax.legend()
    ax.set_yscale('log')

    plt.show()


if __name__ == "__main__":
    plot_cache_hit()
