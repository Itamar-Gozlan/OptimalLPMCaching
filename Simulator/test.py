import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import Utils
from Algorithm import *
import time
import pandas as df

ROOT = 0
ROOT_PREFIX = '0.0.0.0/0'


def random_policy_tree_test():
    condition = False
    # while not condition:
    # policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(10000)]
    with open('../Zipf/prefix_only.txt', 'r') as f:
        policy = f.readlines()
    # policy = {k: v.strip() for k,v in policy.items()}
    # policy = open('../Zipf/prefix_only.txt', 'r').read().splitlines()
    # with open('../Zipf/sorted_prefix_with_weights.json', 'r') as f:
    #     policy_weight = json.load(f)
    res = []
    # print(policy)
    cache_size = 2000
    # policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(10000)]
    OptLPMAlg = OptimalLPMCache(cache_size, policy, dependency_splice=True)

    if '0.0.0.0/0' not in policy:
        policy.append("0.0.0.0/0")

    policy_weight = {k.strip(): np.random.randint(100) for k in policy}
    policy_weight['0.0.0.0/0'] = 0

    t0 = time.time()
    # cache = OptLPMAlg.get_cache(policy_weight)
    print("Not iterative: {0} sec".format(time.time() - t0))
    t0 = time.time()
    iterative_cache = OptLPMAlg.iterative_get_cache(policy_weight)
    print("Iterative: {0} sec".format(time.time() - t0))

    print(cache == iterative_cache)
    res.append(cache == iterative_cache)
    return

    color_map = []
    for v, rule in OptLPMAlg.vertex_to_rule.items():
        if rule in cache:
            color_map.append('red')
            continue
        if rule + "_goto" in cache:
            color_map.append('black')
            continue
        else:
            color_map.append('blue')
            continue
    labels = {v: OptLPMAlg.vertex_to_rule[v] + "\n"
                 + '\n{0}'.format(policy_weight[OptLPMAlg.vertex_to_rule[v]])
              # "\n".join(["{0} : {1} : {2}".format(i, OptLPMAlg.feasible_set[v].feasible_iset[i],
              #                                     OptLPMAlg.feasible_set[v].item_count[i]) for i in
              #            range(cache_size + 1)])
              # "\n {0}".format(OptLPMAlg.feasible_set[v].feasible_iset[cache_size])
              # + "\n vertex: {0}".format(v)
              for v in
              list(OptLPMAlg.policy_tree.nodes)}
    # draw_policy_trie(T, labels, 'optimal cache', x_shift=0, y_shift=-20)

    Utils.draw_tree(OptLPMAlg.policy_tree, labels, x_shift=0, y_shift=10, color_map=color_map)
    plt.show()


class NodeData:
    def __init__(self, weight=0, size=0, distance=0, n_successors=0):
        self.subtree_weight = weight
        self.subtree_size = size
        self.subtree_depth = distance
        self.n_successors = n_successors
        self.v = None

    def unpack(self):
        return self.subtree_size, self.n_successors, self.v


def check_zipf_policy_tree():
    with open('../Zipf/prefix_only.txt', 'r') as f:
        policy = f.readlines()

    policy = list(map(lambda s: s.strip(), policy))
    policy_tree, rule_to_vertex, successors = OptimalLPMCache.process_policy(policy)
    vertex_to_rule = {value: key for key, value in rule_to_vertex.items()}
    depth_dict = OptimalLPMCache.construct_depth_dict(policy_tree)
    # Looking for nodes with low count of successors and big subtrees
    data_dict = {}
    for depth in sorted(list(depth_dict.keys()), reverse=True):
        for v in depth_dict[depth]:
            node_data = NodeData()
            node_data.subtree_size = 1
            if len(successors[v]) == 0:  # leaf
                node_data.subtree_depth = 1
            else:
                for u in successors[v]:
                    node_data.subtree_size += data_dict[u].subtree_size
                node_data.subtree_depth = 1 + max(
                    [data_dict[u].subtree_depth for u in successors[v]])
            node_data.v = v
            node_data.n_successors = len(successors[v])
            data_dict[v] = node_data

    a = 1.8
    n_pkt = 0
    heavy_weight_candidates = [x[-1] for x in
                               sorted([nd.unpack() for nd in data_dict.values()], key=lambda x: x[0], reverse=True)[1:][
                               :5]]
    rest = [x[-1] for x in sorted([nd.unpack() for nd in data_dict.values()], key=lambda x: x[0], reverse=True)[1:][5:]]
    np.random.shuffle(rest)

    with open('../Zipf/sorted_prefix_5.json', 'w') as f:
        json.dump([vertex_to_rule[u] for u in heavy_weight_candidates + rest], f)

    # weights = sorted(np.random.zipf(a, len(policy))-1, reverse=True)
    # pfx_weight = {vertex_to_rule[v] : w for v,w in zip(heavy_weight_candidates + rest, weights)}
    #
    # with open('../Zipf/sorted_prefix_with_weights_5.json', 'w') as f:
    #     json.dump(pfx_weight, f, default=str)
    #
    # print("s")

    # algorithm = OptimalLPMCache(cache_size=10, policy=policy, dependency_splice=True)
    # cache = algorithm.get_cache(pfx_weight)

    print("s")


def playground():
    policy = ['0.0.0.0/0', '192.168.1.0/24', '192.168.0.0/24', '192.168.0.1/32', '192.168.0.0/32', '192.168.1.1/32',
              '192.168.1.0/32']

    with open('../Zipf/prefix_only.txt', 'r') as f:
        policy = f.readlines()

    cache_size = 10
    OptLPMAlg = OptimalLPMCache(cache_size, policy, dependency_splice=True)

    child_histogram_data = sorted([len(OptLPMAlg.successors[v]) for v in OptLPMAlg.successors])

    child_histogram = {}
    for nc in child_histogram_data:
        child_histogram[nc] = 1 + child_histogram.get(nc, 0)

    low = 0
    high = 1
    log_bins = {}
    while low <= sorted(list(child_histogram.keys()))[-2]:
        sum_tot = 0
        key = "[{0},{1})".format(low, high)
        while low <= high - 1:
            sum_tot = child_histogram.get(low, 0)
            low += 1
        log_bins[key] = sum_tot
        low = high
        high *= 10

    fig, ax = plt.subplots()
    # data_x, data_y = zip(*sorted(list(child_histogram.items()), key=lambda x: x[0]))
    ax.plot(list(log_bins.keys())[1:], list(log_bins.values())[1:], label="Count")

    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel("# Children")
    ax.set_ylabel("# Node Count")
    ax.set_title("Number Of Children Per Node \n Log Bin")
    plt.show()
    print(log_bins)

    """



    return
    """

    # head = list(filter(lambda v: 3 <= len(OptLPMAlg.successors[v]) <= 40, OptLPMAlg.successors.keys()))
    # remain = list(set(OptLPMAlg.successors.keys()) - set(head))
    #
    # sorted_zipf_prefix = []
    # for vtx in head + remain:
    #     sorted_zipf_prefix.append(OptLPMAlg.vertex_to_rule[vtx])
    #
    # with open('../Zipf/zipf_sorted_headers.json', 'w') as f:
    #     json.dump(sorted_zipf_prefix, f)
    #
    # data = sorted([len(x) for x in list(filter(bool, list(OptLPMAlg.successors.values())[1:]))], reverse=True)
    #
    # fig, ax = plt.subplots()
    # # data_x, data_y = zip(*sorted(list(child_histogram.items()), key=lambda x: x[0]))
    # ax.plot(list(range(len(data))), data, label="Number of children")
    # ax.plot(list(range(len(data))), [np.average(data)] * len(data), label="Average")
    #
    # ax.set_yscale('log')
    # # ax.set_xscale('log')
    # ax.set_xlabel("Node")
    # ax.set_ylabel("# Children")
    # ax.set_title("Distribution among number of children \n"
    #              "AVG: {0} MAX : {1} MIN :{2}".format(np.average(data), np.max(data), np.min(data)))
    # plt.show()
    #
    # print("Here we go...")
    # t0 = time.time()
    # leaf_children_data = {}
    # T = nx.DiGraph()
    # for depth in sorted(list(OptLPMAlg.depth_dict.keys())):
    #     for root in OptLPMAlg.depth_dict[depth]:
    #         leaf_node = 0
    #         if len(OptLPMAlg.successors[root]) == 0:
    #             continue
    #         for u in OptLPMAlg.successors[root]:
    #             if len(OptLPMAlg.successors[u]) > 0:
    #                 T.add_edge(root, u)
    #             else:
    #                 leaf_node += 1
    #         new_node = max(T.nodes) + 1
    #         T.add_edge(root, new_node)
    #         leaf_children_data[new_node] = leaf_node

    # labels = {v : leaf_children_data.get(v, '') for v in T.nodes}
    # Utils.draw_tree(T, labels)
    # print("saving")
    # plt.set_size_inches(42, 24)
    # plt.savefig('standford_backbone.jpg')

    # plt.show()

    # cache = OptLPMAlg.get_cache(policy_weight)
    # print("Elapsed time: {0}".format(time.time() - t0))
    # print(child_histogram)
    # fig, ax = plt.subplots()
    # data_x, data_y = zip(*sorted(list(child_histogram.items()), key=lambda x: x[0]))
    # ax.plot(data_x, data_y)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlabel("# Children")
    # ax.set_ylabel("# Nodes")
    # ax.set_title("Histogram of children count per node")
    # plt.show()

    #     with open('../Zipf/prefix_with_weights.json', 'r') as f:
    #         data = json.load(f)
    #
    #     shortest_path = nx.single_source_shortest_path(OptLPMAlg.policy_tree, ROOT)
    #     longest_distance = max([len(sp) for sp in shortest_path.values()])
    #     policy_weight = {rule: 16 ** (longest_distance - len(shortest_path[OptLPMAlg.rule_to_vertex[rule]])) for rule in
    #                      policy}
    #
    #     policy_weight['0.0.0.0/0'] = 0
    #
    #     cache = OptLPMAlg.get_cache(policy_weight)
    #     print("cache {0}".format(cache))
    #     condition = True
    # color_map = []
    # for v, rule in OptLPMAlg.vertex_to_rule.items():
    #     if rule in cache:
    #         color_map.append('red')
    #         continue
    #     if rule + "_goto" in cache:
    #         color_map.append('black')
    #         continue
    #     else:
    #         color_map.append('blue')
    #         continue
    # labels = {v: OptLPMAlg.vertex_to_rule[v] + "\n" +
    #              # '\n{0}'.format(policy_weight[OptLPMAlg.vertex_to_rule[v]]) +
    #              "\n".join(["{0} : {1} : {2}".format(i, OptLPMAlg.feasible_set[v].feasible_iset[i],
    #                                                  OptLPMAlg.feasible_set[v].item_count[i]) for i in
    #                         range(cache_size + 1)])
    #              # "\n {0}".format(OptLPMAlg.feasible_set[v].feasible_iset[cache_size])
    #              + "\n vertex: {0}".format(v)
    #           for v in
    #           list(OptLPMAlg.policy_tree.nodes)}
    # # draw_policy_trie(T, labels, 'optimal cache', x_shift=0, y_shift=-20)
    #
    # Utils.draw_tree(OptLPMAlg.policy_tree, labels, x_shift=0, y_shift=10, color_map=color_map)
    # plt.show()


def check_feasible_iset():
    feasible_set_array = []
    cache_size = 10
    for n in range(1000):
        fs = FeasibleSet(cache_size)
        for i in range(cache_size):
            fs.insert_iset_item(i, "i {0}, n {1}".format(i, n), np.random.randint(100))
        feasible_set_array.append(fs)

    X = FeasibleSet.OptDTUnion(feasible_set_array, cache_size)
    # fs1 = FeasibleSet(cache_size)
    # fs2 = FeasibleSet(cache_size)
    #
    # for i in range(cache_size):
    #     fs1.insert_iset(i,1,1)
    #     fs2.insert_iset(i, 2, 2)
    print("s")

def format_result_into_table(path):
    df = pd.DataFrame(columns=["Epoch", "Cache Size %", "Splice", "Hit Rate"])
    row_data = []
    for dirpath in os.listdir(path):
        with open(path + "/" + dirpath, 'r') as f:
            data = f.readlines()
            if len(data) > 1 and "Total Accesses: 33364812" in data[-1]:
                epoch = dirpath.split("epoch")[1].split("_")[0]
                cache_size_p = dirpath.split("cachesizep")[1].split("_")[0]
                splice = dirpath.split("splice")[1].split("_")[0].split('.')[0]
                hit_rate = data[-1].split(":")[3].split(',')[0]
                row = {"Epoch": epoch,
                       "Cache Size %": cache_size_p,
                       "Splice": splice,
                       "Hit Rate": hit_rate}
                df = df.append(row, ignore_index=True)

    df.to_csv('../2507_result.csv')


def main():
    # check_feasible_iset()
    playground()
    # random_policy_tree_test()
    # check_zipf_policy_tree()
    # format_result_into_table("/home/itamar/PycharmProjects/OptimalLPMCaching/2507")

if __name__ == "__main__":
    main()
