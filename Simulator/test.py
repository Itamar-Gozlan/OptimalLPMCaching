import json

import matplotlib.pyplot as plt
import numpy as np
import Utils
from Algorithm import *
import time

policy = ['0001', '0', '000100', '011', '110', '0100001', '10000100', '1110011', '11101', '1', '111', '0101001',
          '0100000', '1000100', '000101']


def random_policy_tree_test():
    condition = False
    cache_size = 10
    while not condition:
        policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(100)]
        print(policy)
        OptLPMAlg = OptimalLPMCache(cache_size, policy, dependency_splice=True)

        if '0.0.0.0/0' not in policy:
            policy.append("0.0.0.0/0")

        shortest_path = nx.single_source_shortest_path(OptLPMAlg.policy_tree, ROOT)
        longest_distance = max([len(sp) for sp in shortest_path.values()])

        policy_weight = {rule: 16 ** (longest_distance - len(shortest_path[OptLPMAlg.rule_to_vertex[rule]])) for rule in
                         policy}
        policy_weight['0.0.0.0/0'] = 0
        cache = OptLPMAlg.get_cache(policy_weight)

        condition = True in list(map(lambda s: 'goto' in s, cache))

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


def main():
    policy = ['0.0.0.0/0', '192.168.1.0/24', '192.168.0.0/24', '192.168.0.1/32', '192.168.0.0/32', '192.168.1.1/32',
              '192.168.1.0/32']

    condition = False
    while not condition:
        # policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(100000)]
        # print(policy)
        with open('../Zipf/prefix_only.txt', 'r') as f:
            policy = f.readlines()

        if '0.0.0.0/0' not in policy:
            policy.append("0.0.0.0/0")

        cache_size = 100
        OptLPMAlg = OptimalLPMCache(cache_size, dependency_splice=False)
        policy_weight = {rule.strip(): 1 for rule in policy}
        print("Here we go...")
        t0 = time.time()

        cache = OptLPMAlg.get_cache(policy_weight)
        print("Elapsed time: {0}".format(time.time() - t0))

        return

        with open('../Zipf/prefix_with_weights.json', 'r') as f:
            data = json.load(f)

        shortest_path = nx.single_source_shortest_path(OptLPMAlg.policy_tree, ROOT)
        longest_distance = max([len(sp) for sp in shortest_path.values()])
        policy_weight = {rule: 16 ** (longest_distance - len(shortest_path[OptLPMAlg.rule_to_vertex[rule]])) for rule in
                         policy}

        policy_weight['0.0.0.0/0'] = 0

        cache = OptLPMAlg.get_cache(policy_weight)
        print("cache {0}".format(cache))
        condition = True
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
    labels = {v: OptLPMAlg.vertex_to_rule[v] + "\n" +
                 # '\n{0}'.format(policy_weight[OptLPMAlg.vertex_to_rule[v]]) +
                 "\n".join(["{0} : {1} : {2}".format(i, OptLPMAlg.feasible_set[v].feasible_iset[i], OptLPMAlg.feasible_set[v].item_count[i]) for i in range(cache_size + 1)])
                 # "\n {0}".format(OptLPMAlg.feasible_set[v].feasible_iset[cache_size])
                 + "\n vertex: {0}".format(v)
                  for v in
              list(OptLPMAlg.policy_tree.nodes)}
    # draw_policy_trie(T, labels, 'optimal cache', x_shift=0, y_shift=-20)

    Utils.draw_tree(OptLPMAlg.policy_tree, labels, x_shift=0, y_shift=10, color_map=color_map)
    plt.show()


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


if __name__ == "__main__":
    # check_feasible_iset()
    # main()
    random_policy_tree_test()
