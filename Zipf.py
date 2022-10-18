import json
import sys
from random import random

from matplotlib import pyplot as plt
import numpy as np
import json
from Algorithm import HeuristicLPMCache
import os





def plot_zipf(weights):
    fig, ax = plt.subplots()
    weights_int = list(map(int, weights.values()))

    # weights_int = [1]*10000
    sum_tot = sum(weights_int)
    # n = 100000

    # print(sum([x/sum_tot for x in weights_int[:n]]))
    # print(np.cumsum([(x * 100) / sum_tot for x in sorted(weights_int, reverse=True)]))
    ax.plot(list(range(len(weights_int))),
            np.cumsum([(x)*100 / sum_tot for x in sorted(weights_int, reverse=True)]), marker="o")

    xy_label_font_size = 28
    ax.xaxis.set_tick_params(labelsize=xy_label_font_size)
    # ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

    ax.set_ylabel('CDF (%)', fontsize=xy_label_font_size)
    ax.set_xlabel("Size", fontsize=xy_label_font_size)
    # ax.set_ylim(ylim)
    # ax.legend(prop=dict(size=13))  # , loc='lower left')
    ax.grid(True)

    # path_to_save = path_to_save.replace('.jpg', '_tls.jpg')
    # path_to_save = 'result/Figures/1510/' + path_to_save.split('/')[-1]
    # plt.show()
    fig.tight_layout()
    # h = 4
    # fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
    # fig.savefig(path_to_save, dpi=300)

    # ax.set_yscale('log')
    # ax.set_title("Weights")
    plt.show()


def generate_30M_weights():
    with open("Zipf/prefix_only.txt", 'r') as f:
        prefix_array = f.readlines()
    # p = "zipf_sorted_headers.json"
    # p = "sorted_prefix_5.json"
    # with open(p, 'r') as f:
    #     prefix_array = json.load(f)

    # a = 1.67
    a = 1.8
    n_pkt = 1
    weights = []
    sum_60 = 0
    while not (30E6 < n_pkt < 35E6 and 0.9 < sum_60):
        weights = np.random.zipf(a, len(prefix_array))
        n_pkt = sum((weights))
        sum_60 = sum(sorted(weights)[-60:]) / n_pkt
        if 30E6 < n_pkt < 35E6:  # or 0.5 < sum_60 < 0.7:
            print("pkt_n: {0} sum_60: {1}".format(n_pkt, sum_60))

    print("+++++++++++")
    print(max(weights) / n_pkt)

    with open("Zipf/zipf_weight_30M_better.json", 'w') as f:
        json.dump(sorted(list(weights), reverse=True), f, default=str)

    # prefix_weight = {}
    # for prefix, zipf_w in zip(prefix_array, sorted(weights, reverse=True)):
    #     prefix_weight[prefix.strip()] = zipf_w
    #
    # plot_zipf(prefix_weight)

    # with open("Zipf/short_prefix_with_weights.json", 'w') as f:
    #     json.dump(prefix_weight, f, default=str)


def calculate_packet_array_and_prefix2weight(zipf_weight, ordered_nodes, vertex_to_rule, trace_name):
    with open(zipf_weight, 'r') as f:
        sorted_zipf_weight = json.load(f)

    prefix_to_weight = {}
    for idx, vtx in enumerate(ordered_nodes):
        prefix_to_weight[vertex_to_rule[vtx]] = sorted_zipf_weight[idx]  # string

    with open(trace_name + "_prefix2weight.json", 'w') as f:
        json.dump(prefix_to_weight, f)

    produce_packet_array_of_prefix_weight(prefix_to_weight, trace_name + "_packet_array.json")


def generate_sorted_traces_by_node_data():
    # with open("Caida/6000rules_small/caida_traceTCP_policy.json", 'r') as f:
    #     # policy = f.readlines()
    #     policy = json.load(f)
    #
    # with open("Caida/6000rules_small/caida_traceTCP_prefix_weight.json", 'r') as f:
    #     # policy = f.readlines()
    #     pfx_w = json.load(f)

    with open("Zipf/prefix_only.txt", 'r') as f:
        policy = f.readlines()

    policy = list(map(lambda st: st.strip(), policy))
    node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(policy)
    candidates = set(node_data_dict.keys())
    candidates.remove(0)
    sorted_by_subtree_size = sorted(candidates, key=lambda node: node_data_dict[node].subtree_size,
                                    reverse=True)
    sorted_by_subtree_size_dvd_n_children = sorted(node_data_dict.keys(),
                                                   key=lambda node: node_data_dict[node].subtree_size / (
                                                               node_data_dict[node].n_successors + 0.01), reverse=True)

    sorted_by_node_depth = sorted(node_data_dict.keys(),
                                                   key=lambda node: node_data_dict[node].subtree_depth , reverse=True)
    # cache_size = 1024
    # optLPM_true = OptimalLPMCache(cache_size=cache_size, policy=policy, dependency_splice=True)
    # cache_true = optLPM_true.get_cache(pfx_w)
    # optLPM_false = OptimalLPMCache(cache_size=cache_size, policy=policy, dependency_splice=False)
    # cache_false = optLPM_false.get_cache(pfx_w)

    zipf_weight = "Zipf/zipf_weight_30M_sum60_70.json"
    # zipf_weight = "Zipf/zipf_weight_30M_sum60_90.json"
    # trace_name = ''.join([zipf_weight.split('.')[0], "_sorted_by_subtree_size", '.json'])
    # calculate_packet_array_and_prefix2weight(zipf_weight, sorted_by_subtree_size, vertex_to_rule, trace_name)

    # trace_name = ''.join([zipf_weight.split('.')[0], "sorted_by_subtree_size_dvd_n_children", '.json'])
    # calculate_packet_array_and_prefix2weight(zipf_weight, sorted_by_subtree_size_dvd_n_children, vertex_to_rule, trace_name)

    # trace_name = ''.join([zipf_weight.split('.')[0], "sorted_by_subtree_size_dvd_n_children", '.json'])
    # calculate_packet_array_and_prefix2weight(zipf_weight, sorted_by_subtree_size_dvd_n_children, vertex_to_rule, trace_name)

    trace_name = ''.join([zipf_weight.split('.')[0], "sorted_by_node_depth", '.json'])
    calculate_packet_array_and_prefix2weight(zipf_weight, sorted_by_node_depth, vertex_to_rule, trace_name)


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

    T, rule_to_vertex, successors = HeuristicLPMCache.process_policy(prefix_array)
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
    with open("traces/caida_traceUDP_prefix_weight.json", 'r') as f:
        prefix_weight = json.load(f)
    prefix_weight = {k : v for k,v in sorted(prefix_weight.items(), key=lambda x: x[1], reverse=True)}
    plot_zipf(prefix_weight)

    # generate_sorted_traces()
    # generate_sorted_traces_by_node_data()
