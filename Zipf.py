import json
import sys
from random import random

from matplotlib import pyplot as plt
import numpy as np
import json
from Algorithm import OptimalLPMCache
import os


def plot_zipf(weights):
    fig, ax = plt.subplots()
    weights_int = list(map(int, weights.values()))
    sum_tot = sum(weights_int)
    n = 100

    # print(sum([x/sum_tot for x in weights_int[:n]]))
    print(np.cumsum([(x * 100) / sum_tot for x in sorted(weights_int, reverse=True)][:n]))
    ax.plot(list(range(len(weights_int)))[:n],
            np.cumsum([(x * 100) / sum_tot for x in sorted(weights_int, reverse=True)][:n]), marker="o")
    # ax.plot(list(range(len(weights_int))), [np.average(sorted(weights_int))]*len(weights_int))

    # ax.set_yscale('log')
    ax.set_title("Weights")
    plt.show()


def generate_30M_weights():
    with open("Zipf/prefix_only.txt", 'r') as f:
        prefix_array = f.readlines()
    # p = "zipf_sorted_headers.json"
    # p = "sorted_prefix_5.json"
    # with open(p, 'r') as f:
    #     prefix_array = json.load(f)

    a = 1.67
    n_pkt = 1
    weights = []
    sum_60 = 0
    while not (30E6 < n_pkt < 35E6 and  0.5 < sum_60 < 0.7):
        weights = np.random.zipf(a, len(prefix_array))
        n_pkt = sum((weights))
        sum_60 = sum(sorted(weights)[-60:])/n_pkt
        if 30E6 < n_pkt < 35E6: #or 0.5 < sum_60 < 0.7:
            print("pkt_n: {0} sum_60: {1}".format(n_pkt, sum_60))

    print("+++++++++++")
    print(max(weights) /n_pkt)

    with open("Zipf/zipf_weight_30M_better.json", 'w') as f:
        json.dump(sorted(list(weights), reverse=True), f, default=str)

    # prefix_weight = {}
    # for prefix, zipf_w in zip(prefix_array, sorted(weights, reverse=True)):
    #     prefix_weight[prefix.strip()] = zipf_w
    #
    # plot_zipf(prefix_weight)

    # with open("Zipf/short_prefix_with_weights.json", 'w') as f:
    #     json.dump(prefix_weight, f, default=str)


def generate_sorted_traces():
    bottom = int(sys.argv[1])
    top = int(sys.argv[2])
    trace_name = "zipf_trace_{0}_{1}".format(bottom, top)
    print(trace_name)
    base_path = 'traces/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    with open("Zipf/prefix_only.txt", 'r') as f:
        prefix_array = f.readlines()
    prefix_array = list(map(lambda st: st.strip(), prefix_array))

    T, rule_to_vertex, successors = OptimalLPMCache.process_policy(prefix_array)
    vertex_to_rule = {v: k for k, v in rule_to_vertex.items()}

    head = list(filter(lambda v: bottom <= len(successors[v]) <= top, successors.keys()))
    print(len(head))
    remain = list(set(successors.keys()) - set(head))
    np.random.shuffle(remain)

    with open("Zipf/zipf_weight_30M_better.json", 'r') as f:
        sorted_zipf_weight = json.load(f)

    prefix_to_weight = {}
    for idx, vtx in enumerate(head + remain):
        prefix_to_weight[vertex_to_rule[vtx]] = sorted_zipf_weight[idx]  # string

    with open(base_path + trace_name + "_prefix2weight.json", 'w') as f:
        json.dump(prefix_to_weight, f)

    produce_packet_array_of_prefix_weight(prefix_to_weight, base_path + trace_name + "_packet_array.json")


def produce_packet_array_of_prefix_weight(prefix_weight, path_to_save):
    packet_array = []
    count = 0
    for key, value in prefix_weight.items():
        for i in range(np.int64(value)):
            packet_array.append(key)
        if len(packet_array) > count:
            count += 100000
            print(len(packet_array))

    np.random.shuffle(packet_array)
    with open(path_to_save, 'w') as f:
        json.dump(packet_array, f, default=str)

    print("s")


if __name__ == "__main__":
    # generate_30M_weights()
    # main()
    # with open("Zipf/prefix_with_weights.json", 'r') as f:
    #     prefix_weight = json.load(f)
    # plot_zipf(prefix_weight)
    generate_sorted_traces()
