import json
from random import random

from matplotlib import pyplot as plt
import numpy as np
import json


def plot_zipf(weights):
    fig, ax = plt.subplots()
    weights_int = list(map(int, weights.values()))
    ax.plot(list(range(len(weights_int))), sorted(weights_int, reverse=True))
    ax.set_yscale('log')
    ax.set_title("Weights")
    plt.show()


def generate_30M_weights():
    with open("short_prefix_only.txt", 'r') as f:
        prefix_array = f.readlines()
    a = 1.8
    n_pkt = 0
    weights = []
    while n_pkt < 30E6 or n_pkt > 35E6:
        weights = np.random.zipf(a, len(prefix_array))
        n_pkt = sum((weights))
        print(n_pkt)

    prefix_weight = {}
    for prefix, zipf_w in zip(prefix_array, weights):
        prefix_weight[prefix.strip()] = zipf_w

    with open("short_prefix_with_weights.json", 'w') as f:
        json.dump(prefix_weight, f, default=str)


def main():
    with open("short_prefix_with_weights.json", 'r') as f:
        prefix_weight = json.load(f)
    packet_array = []
    # print("packets : {0}".format(sum(key=int, list(prefix_weight.values()))))
    count = 0
    for key, value in prefix_weight.items():
        for i in range(int(value)):
            packet_array.append(key)
        if len(packet_array) > count:
            count += 100000
            print(len(packet_array))

    np.random.shuffle(packet_array)
    with open(" .json", 'w') as f:
        json.dump(packet_array[:500000], f, default=str)

    print("s")


if __name__ == "__main__":
    # generate_30M_weights()
    # main()
    with open("prefix_with_weights.json", 'r') as f:
        prefix_weight = json.load(f)
    plot_zipf(prefix_weight)
