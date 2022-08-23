import json
import math
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from Zipf import produce_packet_array_of_prefix_weight, NodeData
import Utils
from Algorithm import *
import time
import pandas as df
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

ROOT = 0
ROOT_PREFIX = '0.0.0.0/0'


def hit_to_miss(ser):
    return list(map(lambda v: 100 - v, ser))


class RunCheck:
    @staticmethod
    def get_random_policy_and_weight(avg_degree, n_nodes):
        curr_depth = 0
        ROOT = ""
        depth_dict = {0: [ROOT]}
        tree_nodes = ['']
        base = avg_degree
        if avg_degree == 1:
            req_depth = 32
        else:
            req_depth = min(32, int(np.ceil(np.log(n_nodes) / np.log(base))))

        n_bits = int((32 / req_depth))
        n_bits = min(31, n_bits)
        while curr_depth < req_depth and len(tree_nodes) < n_nodes:
            for v_rule in depth_dict[curr_depth]:
                if len(tree_nodes) == n_nodes:
                    break
                n_children = avg_degree
                for u in range(n_children):
                    if len(tree_nodes) == n_nodes:
                        break
                    random_ip = np.random.randint(0, 2 ** n_bits)
                    bit_str = "".join(['0'] * (n_bits - len(f'0b{random_ip:b}'.split('b')[-1]))) + \
                              f'0b{random_ip:b}'.split('b')[-1]
                    depth_dict[curr_depth + 1] = [v_rule + bit_str] + depth_dict.get(curr_depth + 1, [])
                    tree_nodes.append(v_rule + bit_str)

            curr_depth += 1

        policy = []

        for depth in depth_dict:
            for v_rule in depth_dict[depth]:
                lpm_rule = Utils.binary_lpm_to_str(v_rule)
                policy.append(lpm_rule)
        return policy

    @staticmethod
    def random_policy_tree_test():
        condition = False
        # while not condition:
        # policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(10000)]
        # policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(10000)]
        OptLPMAlg = HeuristicLPMCache(cache_size, policy, dependency_splice=True)

        if '0.0.0.0/0' not in policy:
            policy.append("0.0.0.0/0")

        policy_weight = {k.strip(): np.random.randint(100) for k in policy}
        policy_weight['0.0.0.0/0'] = 0

        cache = OptLPMAlg.get_cache(policy_weight)

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

        print("s")

    @staticmethod
    def playground():
        policy = ['0.0.0.0/0', '192.168.1.0/24', '192.168.0.0/24', '192.168.0.1/32', '192.168.0.0/32', '192.168.1.1/32',
                  '192.168.1.0/32']

        with open('../Zipf/prefix_only.txt', 'r') as f:
            policy = f.readlines()

        cache_size = 10
        OptLPMAlg = HeuristicLPMCache(cache_size, policy, dependency_splice=True)

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

    @staticmethod
    def check_feasible_iset():
        feasible_set_array = []
        cache_size = 10
        for n in range(1000):
            fs = FeasibleSet(cache_size)
            for i in range(cache_size):
                fs.insert_iset_item(i, "i {0}, n {1}".format(i, n), np.random.randint(100))
            feasible_set_array.append(fs)

        X = FeasibleSet.OptDTUnion(feasible_set_array, cache_size)

    @staticmethod
    def running_example():

        binary_policy = [
            '',
            '00001010',
            '0000101000000001',
            '01000000',
            '0100000000001010',
            '010000000000101000001000',
            '010000000000101000000010',
            '01000000000010100000001000001010'
        ]

        policy = {
            '0.0.0.0/0': 0,
            '10.0.0.0/8': 12,
            '10.1.0.0/16': 17,
            '64.0.0.0/8': 25,
            '64.10.0.0/16': 3,
            '64.10.8.0/24': 5,
            '64.10.2.0/24': 10,
            '64.10.2.10/32': 15
        }

        # policy = {}
        # for b_p in binary_policy:
        #     policy[b_p] = Utils.binary_lpm_to_str(b_p)
        #
        # r_po = {v: k for k, v in policy.items()}
        #
        # print(policy.values())

        cache_size = 4
        algorithm = OptimalLPMCache(policy.keys(), policy, cache_size)
        algorithm.get_optimal_cache()
        # algorithm.to_json('running_example')

        color_map = []
        for vtx in algorithm.policy_tree.nodes:
            # color_map.append('lime')
            if vtx in algorithm.S[ROOT][1][4]:
                color_map.append('pink')
                continue
            if vtx in algorithm.gtc_nodes:
                color_map.append('gray')
                continue
            else:
                color_map.append('lime')

        # labels = {v: str(v) + "\n" + str(prefix_weight[algorithm.vertex_to_rule[v]] + "\n" + algorithm.vertex_to_rule[v])
        #           for idx, v in enumerate(algorithm.policy_tree.nodes)}

        # labels = {v: "{0} \n {1}".format(policy[key], policy[])
        # for idx, v in enumerate(algorithm.policy_tree.nodes)}

        labels = {v: "{0} \n {1} \n {2}".format(algorithm.vertex_to_rule[v], policy[algorithm.vertex_to_rule[v]], v) for
                  v in
                  algorithm.policy_tree.nodes}

        # nx.draw(algorithm.policy_tree, labels=labels)
        Utils.draw_tree(algorithm.policy_tree, labels, color_map=color_map)
        plt.show()

    @staticmethod
    def draw_policy_tree_from_algorithm(optimal_lpm_cache, cache_size, prefix_weight, rule_names=None):
        color_map = []
        for vtx in optimal_lpm_cache.policy_tree.nodes:
            # color_map.append('lime')
            if vtx in optimal_lpm_cache.S[ROOT][1][cache_size]:
                color_map.append('pink')
                continue
            if vtx in optimal_lpm_cache.gtc_nodes:
                color_map.append('gray')
                continue
            else:
                color_map.append('lime')
        if rule_names:
            labels = {v: str(rule_names[optimal_lpm_cache.vertex_to_rule[v]]) + '\n' +
                         str(prefix_weight[optimal_lpm_cache.vertex_to_rule[v]])
                      for idx, v in enumerate(optimal_lpm_cache.policy_tree.nodes)}
        else:
            labels = {v: str(v) + "\n" + str(prefix_weight[optimal_lpm_cache.vertex_to_rule[v]])
                      for idx, v in enumerate(optimal_lpm_cache.policy_tree.nodes)}
        Utils.draw_tree(optimal_lpm_cache.policy_tree, labels, color_map=color_map)
        plt.show()

    @staticmethod
    def test_random_OptLPM():
        avg_degree = 3
        depth = 5
        cache_size = 5
        policy = set(RunCheck.get_random_policy_and_weight(depth, avg_degree))
        zipf_w = np.random.zipf(1.67, len(policy))
        prefix_weight = {p: zipf_w[idx] for idx, p in enumerate(policy)}
        print("policy = {0}".format(policy))
        print("prefix_weight = {0}".format(prefix_weight))
        print("len(policy) = {0}".format(len(policy)))

        prefix_weight[ROOT_PREFIX] = 0

        optimal_lpm_cache = OptimalLPMCache(policy, prefix_weight, cache_size)
        optimized_lpm_cache = OptimizedOptimalLPMCache(policy, prefix_weight, cache_size)

        optimal_lpm_cache.get_optimal_cache()
        optimized_lpm_cache.get_optimal_cache()

        for i in range(cache_size + 1):
            for node in optimal_lpm_cache.vtx_S.keys():
                if optimal_lpm_cache.vtx_S[node][1].get(i) != optimized_lpm_cache.vtx_S[node][1].get(i):
                    print("ERROR!!!! node : {0}".format(node))
                # if optimal_lpm_cache.vtx_S[node][0].get(i) != optimized_lpm_cache.vtx_S[node][0].get(i):
                #     print("node : {0}".format(node))

        RunCheck.draw_policy_tree_from_algorithm(optimal_lpm_cache, cache_size, prefix_weight)

    @staticmethod
    def compare_greedy_opt_solution():
        avg_degree = 2
        depth = 6
        cache_size = 5
        count = 0
        cache_opt = -1
        cache_greedy = -1
        while cache_opt == cache_greedy:
            # cache_size = 4
            #
            # prefix_weight = {
            #     '0.0.0.0/0': 0,
            #     '10.0.0.0/8': 25,
            #     '10.1.0.0/16': 6,
            #     '10.1.1.0/24': 3,
            #     '10.1.1.1/32': 1,
            #     '10.2.0.0/16': 5,
            #     '10.2.2.0/24': 4,
            #     '10.2.2.2/32': 2,  # 1
            # }
            #
            # rule_names = {
            #     '0.0.0.0/0': "R0",
            #     '10.0.0.0/8': "R1",
            #     '10.1.0.0/16': "R2",
            #     '10.1.1.0/24': "R3",
            #     '10.1.1.1/32': "R4",
            #     '10.2.0.0/16': "R5",
            #     '10.2.2.0/24': "R6",
            #     '10.2.2.2/32': "R7",
            # }
            #
            # policy = list(prefix_weight.keys())
            #
            # policy.append(ROOT_PREFIX)
            # prefix_weight[ROOT_PREFIX] = 0

            n_nodes = 100
            degree = 3
            cache_size = 64
            zipf_distribution = np.random.zipf(2.0, n_nodes)
            zipf_distribution = [np.random.randint(3) for i in range(100)]

            node_data_dict, vtx2weight, prefix_weight = SyntheticRho.generate_tree_by_degree(n_nodes, degree,
                                                                                             sorted(
                                                                                                 zipf_distribution))
            rule_names = {rule: "R{0}".format(i) for i, rule in enumerate(prefix_weight.keys())}

            optimal_lpm_cache = OptimalLPMCache(prefix_weight.keys(), prefix_weight, cache_size)
            greedy_lpm_cache = HeuristicLPMCache(cache_size, prefix_weight.keys())
            optimal_lpm_cache.get_optimal_cache()
            cache_opt = optimal_lpm_cache.vtx_S[ROOT][1][cache_size]
            greedy_cache = greedy_lpm_cache.get_cache(prefix_weight)
            cache_greedy = sum([prefix_weight.get(u, 0) for u in greedy_cache])
            print("count: {0}".format(count))
            count += 1
            greedy_gtc = list(map(lambda s: greedy_lpm_cache.rule_to_vertex[s.replace('_goto', '')],
                                  filter(lambda r: 'goto' in r, greedy_cache)))
            greedy_cache_rule = list(map(lambda s: greedy_lpm_cache.rule_to_vertex[s.replace('_goto', '')],
                                         filter(lambda r: 'goto' not in r, greedy_cache)))

            print("===============")
            print(cache_opt)
            print(cache_greedy)
            print("===============")

            RunCheck.draw_policy_tree_from_algorithm(optimal_lpm_cache, cache_size, prefix_weight, rule_names)

            color_map = []
            for vtx in greedy_lpm_cache.policy_tree.nodes:
                # color_map.append('lime')
                if vtx in greedy_cache_rule:
                    color_map.append('pink')
                    continue
                if vtx in greedy_gtc:
                    color_map.append('gray')
                    continue
                else:
                    color_map.append('lime')

            labels = {v: str(rule_names[optimal_lpm_cache.vertex_to_rule[v]]) + '\n' +
                         str(prefix_weight[optimal_lpm_cache.vertex_to_rule[v]])
                      for idx, v in enumerate(optimal_lpm_cache.policy_tree.nodes)}
            Utils.draw_tree(optimal_lpm_cache.policy_tree, labels, color_map=color_map)

            plt.show()

        # print("policy = {0}".format(policy))
        # print("prefix_weight = {0}".format(prefix_weight))
        # print("len(policy) = {0}".format(len(policy)))

    @staticmethod
    def analyze_prefix_weight():
        with open('traces/caida_traceTCP_prefix_weight.json', 'r') as f:
            caida_TCP = json.load(f)
        caida_TCP_node_data_dict, caida_TCP_vertex_to_rule = NodeData.construct_node_data_dict(caida_TCP.keys())
        caida_TCP_rule_to_vertex = {v: k for k, v in caida_TCP_node_data_dict.items()}
        sum_total_caida_tcp = sum(caida_TCP.values())
        caida_tcp_nd_prefix2weight = lambda vtx: caida_TCP[caida_TCP_vertex_to_rule[vtx]] / sum_total_caida_tcp

        with open('traces/caida_traceUDP_prefix_weight.json', 'r') as f:
            caida_UDP = json.load(f)
        caida_UDP_node_data_dict, caida_UDP_vertex_to_rule = NodeData.construct_node_data_dict(caida_UDP.keys())
        caida_UDP_rule_to_vertex = {v: k for k, v in caida_UDP_node_data_dict.items()}
        sum_total_caida_udp = sum(caida_UDP.values())
        caida_udp_prefix2weight = lambda vtx: caida_UDP[caida_UDP_vertex_to_rule[vtx]] / sum_total_caida_udp

        with open('traces/prefix2weight_sum60_70sorted_by_node_depth.json', 'r') as f:
            sorted_prefix2weight = json.load(f)
        stanford_node_data_dict, stanford_vertex_to_rule = NodeData.construct_node_data_dict(
            sorted_prefix2weight.keys())
        standord_rule_to_vertex = {v: k for k, v in stanford_node_data_dict.items()}
        sum_total_sorted_standford = int(sum(map(int, sorted_prefix2weight.values())))
        sorted_zipf_prefix2weight = lambda vtx: int(
            sorted_prefix2weight[stanford_vertex_to_rule[vtx]]) / sum_total_sorted_standford

        with open('traces/zipf_trace_1_0_prefix2weight.json', 'r') as f:
            random_prefix2weight = json.load(f)
        standord_rule_to_vertex = {v: k for k, v in stanford_node_data_dict.items()}
        sum_total_random_standford = int(sum(map(int, sorted_prefix2weight.values())))
        random_zipf_prefix2weight = lambda vtx: int(
            random_prefix2weight[stanford_vertex_to_rule[vtx]]) / sum_total_random_standford

        node_data_tuple_array = [(caida_TCP_node_data_dict, caida_tcp_nd_prefix2weight, "TCP"),
                                 (caida_UDP_node_data_dict, caida_udp_prefix2weight, "UDP"),
                                 (stanford_node_data_dict, sorted_zipf_prefix2weight, "zipf_by_node_depth"),
                                 (stanford_node_data_dict, random_zipf_prefix2weight, "zipf_random")
                                 ]

        no_root_leafs_filter = lambda data_dict: filter(lambda v: v[0] != 0 and v[1].n_successors > 0,
                                                        data_dict.items())
        no_root_filter = lambda data_dict: filter(lambda v: v[0] != 0, data_dict.items())
        avg_feature = lambda data_dict, feature, filter_to_apply: np.average([getattr(nd, feature) for nd in [
            x[1] for x in list(filter_to_apply(data_dict))]])

        avg_feature_traffic_p = lambda data_dict, feature, prefix2weight, filter_to_apply: np.average(
            [getattr(nd, feature) * prefix2weight(nd.v) for nd in [x[1] for x in
                                                                   list(filter_to_apply(data_dict))]])

        avg_feature = lambda data_dict, feature, filter_to_apply: np.average([getattr(nd, feature) for nd in [
            x[1] for x in list(filter_to_apply(data_dict))]])

        calc_rho = lambda data_dict, prefix2weight, filter_to_apply: sum(
            [prefix2weight(nd.v) * ((nd.subtree_size + 1) / (1 + nd.n_successors)) for nd in [
                x[1] for x in list(filter_to_apply(data_dict))]])

        df = pd.DataFrame(columns=["policy",
                                   "subtree_size",
                                   "subtree_size*traffic_p",
                                   "subtree_depth",
                                   "subtree_depth*traffic_p",
                                   "n_successors",
                                   "n_successors*traffic_p",
                                   "nl_subtree_size",
                                   "nl_subtree_size*traffic_p",
                                   "nl_subtree_depth",
                                   "nl_subtree_depth*traffic_p",
                                   "nl_n_successors",
                                   "nl_n_successors*traffic_p",
                                   "rho"
                                   ])
        for data_dict, prefix2weight, policy in node_data_tuple_array:
            row = {
                "policy": policy,
                "subtree_size": avg_feature(data_dict, "subtree_size", no_root_filter),
                "subtree_size*traffic_p": avg_feature_traffic_p(data_dict, "subtree_size", prefix2weight,
                                                                no_root_filter),
                "subtree_depth": avg_feature(data_dict, "subtree_depth", no_root_filter),
                "subtree_depth*traffic_p": avg_feature_traffic_p(data_dict, "subtree_depth", prefix2weight,
                                                                 no_root_filter),
                "n_successors": avg_feature(data_dict, "n_successors", no_root_filter),
                "n_successors*traffic_p": avg_feature_traffic_p(data_dict, "n_successors", prefix2weight,
                                                                no_root_filter),

                "nl_subtree_size": avg_feature(data_dict, "subtree_size", no_root_leafs_filter),
                "nl_subtree_size*traffic_p": avg_feature_traffic_p(data_dict, "subtree_size", prefix2weight,
                                                                   no_root_leafs_filter),
                "nl_subtree_depth": avg_feature(data_dict, "subtree_depth", no_root_leafs_filter),
                "nl_subtree_depth*traffic_p": avg_feature_traffic_p(data_dict, "subtree_depth", prefix2weight,
                                                                    no_root_leafs_filter),
                "nl_n_successors": avg_feature(data_dict, "n_successors", no_root_leafs_filter),
                "nl_n_successors*traffic_p": avg_feature_traffic_p(data_dict, "n_successors", prefix2weight,
                                                                   no_root_leafs_filter),
                "rho": calc_rho(data_dict, prefix2weight, no_root_leafs_filter)
            }
            df = df.append(row, ignore_index=True)
        df.to_csv("policy_features.csv")


class ResultToTable:
    @staticmethod
    def format_special_sort(path):
        df = pd.DataFrame(columns=["Zipf", "Cache Size", "Splice", "Hit Rate"])
        missing_result_array = []
        for dirpath in list(filter(lambda d: 'out' in d, os.listdir(path))):
            with open(path + "/" + dirpath, 'r') as f:
                data = f.readlines()
                if len(data) > 1 and "Hit rate: " in data[-1]:
                    # special sort
                    cache_size = dirpath.split('_')[1]
                    dependency = dirpath.split('_')[2]
                    zipf_distribution = '_'.join(dirpath.split('_')[3:5])

                    row = {"Zipf": zipf_distribution,
                           "Cache Size": cache_size,
                           "Splice": dependency,
                           'Hit Rate': data[-1].split(" ")[-1]}

                    df = df.append(row, ignore_index=True)
                else:
                    print("Missing: {0}".format(dirpath))
                    missing_result_array.append(dirpath)
        print(path + '/' + path.split('/')[-1] + '.csv')
        df.to_csv(path + '/' + path.split('/')[-1] + '.csv', index=False)

    @staticmethod
    def format_result_into_table(path):
        df = pd.DataFrame(columns=["Bottom", "Top", "Zipf", "Cache Size", "Splice", "Hit Rate"])
        missing_result_array = []
        for dirpath in os.listdir(path):
            with open(path + "/" + dirpath, 'r') as f:
                data = f.readlines()
                if len(data) > 1 and "Hit rate: " in data[-1]:
                    # range name
                    # bottom = dirpath.split("_")[3]
                    # top = dirpath.split("_")[4]
                    # cache_size = dirpath.split("_")[5]
                    # dependency = dirpath.split("_")[6].split('.')[0]

                    # row = {"Bottom": bottom,
                    #        "Top": top,
                    #        "Cache Size": cache_size,
                    #        "Splice": dependency,
                    #        'Hit Rate': data[-1].split(" ")[-1]}

                    # special sort
                    # cache_size = dirpath.split('_')[1]
                    # dependency = dirpath.split('_')[2]
                    # zipf_distribution = '_'.join(dirpath.split('_')[3:5])
                    #
                    # row = {"Zipf" : zipf_distribution,
                    #        "Cache Size": cache_size,
                    #        "Splice": dependency,
                    #        'Hit Rate': data[-1].split(" ")[-1]}

                    # caida
                    cache_size = dirpath.split('_')[-2]
                    dependency = dirpath.split('_')[-1].replace('.out', '')
                    dependent_rule = dirpath.split('_')[-3].replace('trace', '')

                    row = {"Dependent Rule": dependent_rule,
                           "Cache Size": cache_size,
                           "Splice": dependency,
                           'Hit Rate': data[-1].split(" ")[-1]}

                    df = df.append(row, ignore_index=True)
                # else:
                #     print("Missing: {0}".format(dirpath))
                #     missing_result_array.append(dirpath)

        # for missing_result in missing_result_array:
        #     b, t, cs, opt = missing_result.split('_')[3:]
        #     opt = opt.split('.')[0]
        #     cmd = "nohup python ../simulator_main.py ../traces/zipf_trace_{0}_{1}_prefix2weight.json ../traces/zipf_trace_{0}_{1}_packet_array.json {2} {3} > offline_zipf_trace_{0}_{1}_{2}_{3}.out &".format(
        #         b, t, cs, opt)
        #     print(cmd)
        print(path + '/' + path.split('/')[-1] + '.csv')
        df.to_csv(path + '/' + path.split('/')[-1] + '.csv')

    @staticmethod
    def join_all_results(greedy_local_csv, opt_csv_path):
        res_df = pd.read_csv(greedy_local_csv)
        true_df = res_df[(res_df['Splice'] == True)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
        false_df = res_df[(res_df['Splice'] == False)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
        cache_size_array = list(false_df['Cache Size'])
        df_opt_raw = pd.read_csv(opt_csv_path)
        df_opt = df_opt_raw[(df_opt_raw['Cache Size'].isin(cache_size_array))]

        df = pd.DataFrame(columns=["Cache Size", "OptLocal", "GreedySplice", "OptSplice"])
        print("s")

        for cache_size in cache_size_array:
            row = {
                "Cache Size": cache_size,
                "OptLocal": float(false_df[(false_df['Cache Size'] == cache_size)]["Hit Rate"]),
                "GreedySplice": float(true_df[(true_df['Cache Size'] == cache_size)]["Hit Rate"]),
                "OptSplice": float(df_opt[(df_opt['Cache Size'] == cache_size)]["Hit Rate"]),
                "GreedySplice - OptLocal": float(true_df[(true_df['Cache Size'] == cache_size)]["Hit Rate"]) - float(
                    false_df[(false_df['Cache Size'] == cache_size)]["Hit Rate"]),
                "OptSplice - GreedySplice": float(df_opt[(df_opt['Cache Size'] == cache_size)]["Hit Rate"]) - float(
                    true_df[(true_df['Cache Size'] == cache_size)]["Hit Rate"]),
                "OptSplice - OptLocal": float(df_opt[(df_opt['Cache Size'] == cache_size)]["Hit Rate"]) - float(
                    false_df[(false_df['Cache Size'] == cache_size)]["Hit Rate"])
            }

            df = df.append(row, ignore_index=True)

        path = "/".join(opt_csv_path.split('/')[:-1]) + "/result_summary.csv"
        print(path)
        df.to_csv(path)


class PlotResultTable:
    @staticmethod
    def plot_true_false_opt_df(true_df, false_df, opt_df, path_to_save, ylim=[0, 100]):
        fig, ax = plt.subplots()
        ax.plot(list(map(str, true_df['Cache Size'])), list(map(lambda v: 100 - v, true_df['Hit Rate'])), marker="s",
                markersize=14, label="GreedySplice")
        ax.plot(list(map(str, false_df['Cache Size'])), list(map(lambda v: 100 - v, false_df['Hit Rate'])), marker="o",
                markersize=14, label="OptLocal")
        ax.plot(list(map(lambda x: str(int(x)), opt_df['Cache Size'])),
                list(map(lambda v: 100 - v, opt_df['Hit Rate'])), marker="P",
                markersize=14, label="OptSplice")

        xy_label_font_size = 28
        ax.xaxis.set_tick_params(labelsize=xy_label_font_size)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_ylabel('Cache Miss (%)', fontsize=xy_label_font_size)
        ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
        ax.set_ylim(ylim)
        ax.legend(prop=dict(size=16))
        ax.grid(True)

        fig.tight_layout()
        print(path_to_save)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)

    @staticmethod
    def plot_range_result_table(greedy_local_csv_path, opt_csv_path):
        df = pd.read_csv(greedy_local_csv_path)
        cache_size_array = sorted(list(df['Cache Size'].drop_duplicates()))
        for index, row in df[['Bottom', 'Top']].drop_duplicates().iterrows():
            top = row['Top']
            bottom = row['Bottom']
            res_df = df[(df['Bottom'] == bottom) &
                        (df['Top'] == top)]
            true_df = res_df[(res_df['Splice'] == True)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
            false_df = res_df[(res_df['Splice'] == False)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')

        df_opt = pd.read_csv(opt_csv_path)

        PlotResultTable.plot_true_false_df(true_df, false_df, greedy_local_csv_path.replace('csv', 'jpg'))

        print("s")

    @staticmethod
    def plot_special_sort_result_table(csv_path, opt_csv_path):
        df = pd.read_csv(csv_path)

        true_df = df[(df['Splice'] == True)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
        false_df = df[(df['Splice'] == False)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
        cache_size_array = list(false_df['Cache Size'])
        df_opt_raw = pd.read_csv(opt_csv_path)
        df_opt = df_opt_raw[(df_opt_raw['Cache Size'].isin(cache_size_array))]

        PlotResultTable.plot_true_false_opt_df(true_df, false_df, df_opt, csv_path.replace('csv', 'jpg'))

    @staticmethod
    def calculate_and_save_bar_data(policy_json_path, prefix2weight_json_path, height2weight_histogram_json_path):
        with open(policy_json_path, 'r') as f:
            policy = json.load(f)

        with open(prefix2weight_json_path, 'r') as f:
            prefix2weight = json.load(f)

        node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(policy)

        height2weight_histogram = {}
        count_node_in_depth = {}
        for node in node_data_dict.values():
            weight = prefix2weight[vertex_to_rule[node.v]]
            # height2weight_histogram[node.subtree_depth] = int(weight) + height2weight_histogram.get(node.subtree_depth, 0)
            height2weight_histogram[node.subtree_depth] = node.subtree_size + height2weight_histogram.get(
                node.subtree_depth, 0)
            count_node_in_depth[node.subtree_depth] = 1 + count_node_in_depth.get(node.subtree_depth, 0)

        average_tree_size_for_depth = {depth: total_size / count_node_in_depth[depth] for depth, total_size in
                                       height2weight_histogram.items()}
        with open(height2weight_histogram_json_path, 'w') as f:
            json.dump(average_tree_size_for_depth, f)

    @staticmethod
    def plot_bar(x_data, y_data, path_to_save, y_label, x_label):
        fig, ax = plt.subplots()
        ax.bar(x_data, y_data)
        xy_label_font_size = 25
        ax.xaxis.set_tick_params(labelsize=xy_label_font_size)
        # ax.set_yticks([1, 2, 3, 4, 5])
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_ylabel(y_label, fontsize=xy_label_font_size)
        ax.set_xlabel(x_label, fontsize=xy_label_font_size)
        # ax.set_yscale('yscale', basey=2)

        # ax.set_ylim([1, 300])
        # ax.legend(prop=dict(size=24))
        ax.grid(True)

        fig.tight_layout()
        print(path_to_save)

        # fig.set_size_inches(magic_height * w / (h * dpi), magic_height / dpi)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)

    @staticmethod
    def plot_weight_bar(height2weight_histogram_json_path, path_to_save):
        with open(height2weight_histogram_json_path, 'r') as f:
            height2weight_histogram = json.load(f)

        sum_tot = sum(height2weight_histogram.values())
        x_data = list(map(lambda x: str(int(x) - 1), list(height2weight_histogram.keys())))
        # y_data = [x * 100 / sum_tot for x in list(height2weight_histogram.values())]
        y_data = list(height2weight_histogram.values())
        PlotResultTable.plot_bar(x_data[:-1], y_data[:-1], path_to_save)

    @staticmethod
    def calculate_and_save_bin_bar_data_subtree_size(policy_json_path, path2save):
        with open(policy_json_path, 'r') as f:
            policy = json.load(f)

        node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(policy)
        height2weight_histogram = {}
        for node in node_data_dict.values():
            subtree_size = node.subtree_size
            height2weight_histogram[node.subtree_depth] = subtree_size + height2weight_histogram.get(node.subtree_depth,
                                                                                                     0)
        print("Saving: {0}".format(path2save))
        with open(path2save, 'w') as f:
            json.dump(height2weight_histogram, f)

    @staticmethod
    def plot_subtree_bar(subtree_size_histrogram_json_path, path_to_save_fig):
        with open(subtree_size_histrogram_json_path, 'r') as f:
            st_hist = json.load(f)
        low = 1
        high = 2
        # range [1,2)
        bin_hist = {}
        for subtree_size, count in sorted(st_hist.items(), key=lambda x: x[0]):
            while int(subtree_size) >= high:
                low = high
                high *= 2
            bin_hist[(low, high)] = count + bin_hist.get((low, high), 0)

        print("s {0}".format(sys._getframe(1).f_lineno))
        x_data = list(map(lambda b_t: str(b_t).replace('(', '['), bin_hist.keys()))
        y_data = list(bin_hist.values())
        PlotResultTable.plot_bar(x_data, y_data, path_to_save_fig)

    @staticmethod
    def plot_heatmap(prefix_weight_json, path_to_save=None):
        with open(prefix_weight_json, 'r') as f:
            prefix_weight = json.load(f)
        prefix_weight = {k: int(v) for k, v in prefix_weight.items()}
        policy = list(prefix_weight.keys())
        node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(policy)
        rule_to_vertex = {v: k for k, v in vertex_to_rule.items()}

        bin_idx = 0
        log2_y = [0.5, 0.75, 1]
        node_height = {v: node_data_dict[v].subtree_depth for v in node_data_dict}
        sum_total = sum(prefix_weight.values())
        heatmap_data = {}
        traffic_p = 0
        for prefix, weight in sorted(prefix_weight.items(), key=lambda key: key[1]):
            height = node_height.get(rule_to_vertex[prefix], 0)
            traffic_p += weight / sum_total
            traffic_p = min(traffic_p, 1)
            while traffic_p > log2_y[bin_idx]:
                bin_idx += 1
            # found rule with weight in bin
            heatmap_data[(log2_y[bin_idx], height)] = 1 + heatmap_data.get((log2_y[bin_idx], height), 0)
        p_heatmap_data = {k: 100 * v / len(prefix_weight) for k, v in heatmap_data.items()}
        n_rows = len(log2_y)
        n_cols = len(set(node_height.values()))
        heatmap_array = np.zeros((n_rows, n_cols))
        for (row, col), val in p_heatmap_data.items():  # heatmap_data.items()
            heatmap_array[log2_y.index(row), col - 1] += val

        if path_to_save:
            np.save(path_to_save, heatmap_array)

        # heatmap_array = np.load(path_to_save + '.npy')

        ax = sns.heatmap(heatmap_array, norm=LogNorm(), vmin=1)
        xy_label_font_size = 10
        ax.xaxis.set_tick_params(labelsize=xy_label_font_size)
        # ax.set_yticks(list(map(lambda v : str(v), log2_y)))
        ax.set_yticklabels(log2_y)
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_ylabel('Traffic (%)', fontsize=xy_label_font_size)
        ax.set_xlabel("Subtree height", fontsize=xy_label_font_size)
        # ax.set_yscale('log')
        # ax.set_ylim(ylim)
        # ax.legend(prop=dict(size=24))
        # ax.grid(True)

        plt.show()

        print("s")

        # calculate % traffic per each subtree heught

    @staticmethod
    def plot_result_summary_diff(result_summary_csv):
        df = pd.read_csv(result_summary_csv)
        fig, ax = plt.subplots()
        # ax.plot(list(map(lambda x: str(int(x)), df['Cache Size'])), df["OptSplice - OptLocal"], marker="s",
        #         markersize=14, label="OptSplice - OptLocal")
        # ax.plot(list(map(lambda x: str(int(x)), df['Cache Size'])), df["GreedySplice - OptLocal"], marker="o",
        #         markersize=14, label="GreedySplice - OptLocal")
        ax.plot(list(map(lambda x: str(int(x)), df['Cache Size'])), df["OptSplice - GreedySplice"], marker="P",
                markersize=14, label="OptSplice - GreedySplice")

        xy_label_font_size = 28
        ax.xaxis.set_tick_params(labelsize=xy_label_font_size)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_ylabel('Cache Hit Diff (%)', fontsize=xy_label_font_size)
        ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
        # ax.set_ylim(ylim)
        ax.legend(prop=dict(size=16))
        ax.grid(True)

        fig.tight_layout()
        path_to_save = result_summary_csv.replace('result_summary.csv',
                                                  result_summary_csv.split('/')[-2] + 'greedy_vs_opt.png')
        print(path_to_save)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)


class SyntheticRho:
    @staticmethod
    def generate_binary_strings(bit_count):
        binary_strings = []

        def genbin(n, bs=''):
            if len(bs) == n:
                binary_strings.append(bs)
            else:
                genbin(n, bs + '0')
                genbin(n, bs + '1')

        genbin(bit_count)
        return binary_strings

    @staticmethod
    def construct_d_regular_tree(n_nodes, d, zipf_distribution):
        T = nx.balanced_tree(d, int(math.ceil(math.log(n_nodes, d))), create_using=nx.DiGraph)  # r**h nodes
        nodes_to_bin_string = {0: ""}
        bit_count = np.ceil(math.log(d, 2))
        bin_string_array = SyntheticRho.generate_binary_strings(bit_count)

        prefix2weight = {}
        queue = [ROOT]
        z_idx = 0
        while bool(queue):
            v = queue.pop(0)
            for idx, u in enumerate(T.neighbors(v)):
                nodes_to_bin_string[u] = nodes_to_bin_string[v] + bin_string_array[idx]
                prefix2weight[Utils.binary_lpm_to_str(nodes_to_bin_string[u])] = zipf_distribution[z_idx]
                z_idx += 1
                queue.append(u)
                if len(nodes_to_bin_string) == n_nodes:
                    queue = []
                    break

        prefix2weight['0.0.0.0/0'] = 0
        policy = prefix2weight.keys()
        node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(policy)
        rule_to_vertex = {v: k for k, v in vertex_to_rule.items()}
        vtx2weight = {rule_to_vertex[p]: w for p, w in prefix2weight.items()}

        return node_data_dict, vtx2weight, prefix2weight

        return prefix2weight

    @staticmethod
    def generate_policy(n_nodes, zipf_distribution, percentage):
        dataset_sorted = []
        dataset_reversed = []
        dataset_random = []
        for degree in [2, 4, 8, 16, 32]:
            node_data_dict, vtx2weight, prefix2weight = SyntheticRho.construct_d_regular_tree(n_nodes, degree,
                                                                                              sorted(
                                                                                                  zipf_distribution))
            dataset_sorted.append((node_data_dict, vtx2weight, prefix2weight, degree))
            # node_data_dict, vtx2weight, prefix2weight
            node_data_dict, vtx2weight, prefix2weight = SyntheticRho.construct_d_regular_tree(n_nodes, degree,
                                                                                              sorted(
                                                                                                  zipf_distribution,
                                                                                                  reverse=True))
            dataset_reversed.append((node_data_dict, vtx2weight, prefix2weight, degree))

            node_data_dict, vtx2weight, prefix2weight = SyntheticRho.construct_d_regular_tree(n_nodes, degree,
                                                                                              zipf_distribution)
            dataset_random.append((node_data_dict, vtx2weight, prefix2weight, degree))

        return dataset_sorted, dataset_reversed, dataset_random

    @staticmethod
    def compute_algorithm_results(cache_size, dataset):
        df = pd.DataFrame(columns=["Cache Size", "opt_splice_result",
                                   "greedy_splice_result", "opt_local_result", "degree", "rho", "new_rho"])
        for node_data_dict, vtx2weight, prefix_weight, degree in dataset:
            opt_splice = OptimizedOptimalLPMCache(prefix_weight.keys(), prefix_weight, cache_size)
            # opt_splice = OptimalLPMCache(prefix_weight.keys(), prefix_weight, cache_size_array[-1])
            opt_splice.get_optimal_cache()
            greedy_splice = HeuristicLPMCache(cache_size, prefix_weight.keys(), dependency_splice=True)
            greedy_splice.get_cache(prefix_weight)
            opt_local = HeuristicLPMCache(cache_size, prefix_weight.keys(), dependency_splice=False)
            opt_local.get_cache(prefix_weight)

            sum_total = sum(prefix_weight.values())
            to_hit_rate = lambda val: val * 100 / sum_total

            for i in range(cache_size + 1):
                opt_splice_result = opt_splice.vtx_S[ROOT][1][i]
                greedy_splice_result = greedy_splice.feasible_set[ROOT].feasible_iset_weight[i]
                opt_local_result = opt_local.feasible_set[ROOT].feasible_iset_weight[i]
                row = {"Cache Size": i,
                       "opt_splice_result": to_hit_rate(opt_splice_result),
                       "greedy_splice_result": to_hit_rate(greedy_splice_result),
                       "opt_local_result": to_hit_rate(opt_local_result),
                       "degree": degree,
                       'rho': SyntheticRho.calc_rho(node_data_dict, vtx2weight),
                       "new_rho": SyntheticRho.calc_new_rho(node_data_dict, vtx2weight)
                       }
                df = df.append(row, ignore_index=True)
        return df

    @staticmethod
    def create_zipf(n_nodes, base_dir):
        a = 1.7  # first 60 are 70% of the traffic
        sum_60_70 = -1
        while np.around(sum_60_70, 2) != 0.70:
            zipf_distribution = np.random.zipf(a, n_nodes)
            sum_60_70 = sum(sorted(zipf_distribution, reverse=True)[:60]) / sum(zipf_distribution)
            print(sum_60_70)

        with open(base_dir + '/zipf_distribution_sum60_70.json', 'w') as f:
            json.dump(list(map(int, zipf_distribution)), f)

    @staticmethod
    def generate_rho_experiment(base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(base_dir)
        n_nodes = 6001
        cache_size = 1024

        with open(base_dir + '/zipf_distribution_sum60_70.json', 'r') as f:
            zipf_distribution = list(map(int, json.load(f)))

        dataset_sorted_descending_depth, dataset_sorted_ascending_depth, \
        dataset_random = SyntheticRho.generate_policy(n_nodes, zipf_distribution)
        val = int(sys.argv[1])
        if val == 0:
            df = SyntheticRho.compute_algorithm_results(cache_size, dataset_sorted_descending_depth)
            df.to_csv(base_dir + '/dataset_sorted_descending_depth.csv')
        if val == 1:
            df = SyntheticRho.compute_algorithm_results(cache_size, dataset_random)
            df.to_csv(base_dir + '/dataset_random.csv')
        if val == 2:
            df = SyntheticRho.compute_algorithm_results(cache_size, dataset_sorted_ascending_depth)
            df.to_csv(base_dir + '/dataset_sorted_ascending_depth.csv')

    @staticmethod
    def generate_rho_experiment_percentage(base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(base_dir)
        n_nodes = 6001
        cache_size = 1024

        with open(base_dir + '/zipf_distribution_sum60_70.json', 'r') as f:
            zipf_distribution = list(map(int, json.load(f)))

        p = val = int(sys.argv[2])
        dataset_sorted_descending_depth, dataset_sorted_ascending_depth, \
        dataset_random = SyntheticRho.generate_policy(n_nodes, zipf_distribution, p)
        val = int(sys.argv[1])
        if val == 0:
            df = SyntheticRho.compute_algorithm_results(cache_size, dataset_sorted_descending_depth)
            df.to_csv(base_dir + '/dataset_sorted_descending_depth.csv')
        if val == 1:
            df = SyntheticRho.compute_algorithm_results(cache_size, dataset_random)
            df.to_csv(base_dir + '/dataset_random.csv')
        if val == 2:
            df = SyntheticRho.compute_algorithm_results(cache_size, dataset_sorted_ascending_depth)
            df.to_csv(base_dir + '/dataset_sorted_ascending_depth.csv')

    @staticmethod
    def generate_tree_by_degree(n_nodes, degree, zipf_distribution):
        policy = set()
        break_loop = 0
        while len(policy) < n_nodes:
            policy_addition = RunCheck.get_random_policy_and_weight(degree, n_nodes - len(set(policy)))
            RunCheck.get_random_policy_and_weight(degree, n_nodes - len(set(policy)))
            policy = policy.union(set(policy_addition))
            if n_nodes - len(set(policy)) <= degree:
                break  # can't generate more random subtrees
            break_loop += 1
            if break_loop > 100:
                break

        while len(set(policy)) < n_nodes:  # add leaves to complete to n_nodes
            n_bits = 31
            random_ip = np.random.randint(0, 2 ** n_bits)
            bit_str = "".join(['0'] * (n_bits - len(f'0b{random_ip:b}'.split('b')[-1]))) + \
                      f'0b{random_ip:b}'.split('b')[-1]
            policy.add(Utils.binary_lpm_to_str(bit_str))
        # policy_tree, rule_to_vertex, successors = HeuristicLPMCache.process_policy(policy)
        policy = set(policy)
        node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(policy)
        prefix2weight = {v: w for v, w in zip(sorted(policy, key=lambda p: int(p.split('/')[-1])), zipf_distribution)}
        prefix2weight[ROOT_PREFIX] = 0
        rule_to_vertex = {v: k for k, v in vertex_to_rule.items()}
        vtx2weight = {rule_to_vertex[p]: w for p, w in prefix2weight.items()}

        return node_data_dict, vtx2weight, prefix2weight

    @staticmethod
    def calc_rho(data_dict, vertex2weight):

        no_root_filter = lambda data_dict: filter(lambda v: v[0] != 0,
                                                  data_dict.items())
        sum_total = sum(vertex2weight.values())
        old_rho = sum([(vertex2weight[v] / sum_total) * ((nd.subtree_size + 1) / (1 + nd.n_successors)) for v, nd in
                       list(no_root_filter(data_dict))])
        return old_rho

    @staticmethod
    def calc_new_rho(data_dict, vertex2weight):
        no_root_filter = lambda data_dict: filter(lambda v: v[0] != 0,
                                                  data_dict.items())
        sum_total = sum(vertex2weight.values())

        new_rho = sum([(vertex2weight[v] / sum_total) * (nd.subtree_size + 1) for v, nd in
                       list(no_root_filter(data_dict))]) / sum(
            [(vertex2weight[v] / sum_total) * (nd.n_successors + 1) for v, nd in
             list(no_root_filter(data_dict))])
        return new_rho

    @staticmethod
    def plot_csv_rho(csv_path, rho_type='rho', cache_size=None):
        df = pd.read_csv(csv_path)
        if not cache_size:
            cache_size = max(df['Cache Size'])
        df = df[(df['Cache Size'] == cache_size)]

        df = df.sort_values(by=['degree'], ascending=[False])

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        x_data = list(map(lambda v: str(int(v)), df['degree']))
        # ax1.plot(x_data, hit_to_miss(df['greedy_splice_result']), label="GreedySplice", marker="s")
        ax1.plot(x_data, hit_to_miss(df['opt_local_result']), label="OptLocal", marker="o")
        ax1.plot(x_data, hit_to_miss(df['opt_splice_result']), label="OptSplice", marker="P")
        ax2.plot(x_data, df[rho_type], label=r'$\rho$', marker="d", color='red')
        print(df[rho_type])

        xy_label_font_size = 28
        ax1.xaxis.set_tick_params(labelsize=xy_label_font_size)
        ax2.xaxis.set_tick_params(labelsize=xy_label_font_size)
        ax1.yaxis.set_tick_params(labelsize=xy_label_font_size)
        ax2.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax1.set_xlabel("degree", fontsize=xy_label_font_size)
        ax1.set_ylabel("Cache Miss (%)", fontsize=xy_label_font_size)
        ax2.set_ylabel(r'$\rho$', fontsize=xy_label_font_size)

        ax2.set_ylim([1, 350])
        ax1.set_ylim([0, 100])
        ax2.set_yscale('log')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines = lines_1 + lines_2
        labels = labels_1 + labels_2

        ax1.legend(lines, labels, prop=dict(size=12))
        ax1.grid(True)
        fig.tight_layout()

        # plt.show()

        if rho_type == 'new_rho':
            path_to_save = 'result/rho/sum60_70/Figures/' + csv_path.split('/')[-1].replace('.csv',
                                                                                            '_{0}_new_rho.png'.format(
                                                                                                int(cache_size)))
        else:
            path_to_save = 'result/rho/sum60_70/Figures/' + csv_path.split('/')[-1].replace('.csv',
                                                                                            '_{0}.png'.format(
                                                                                                int(cache_size)))
        print(path_to_save)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)

    @staticmethod
    def plot_csv_rho_diff(csv_path, rho_type='rho', cache_size=None):
        df = pd.read_csv(csv_path)
        if not cache_size:
            cache_size = max(df['Cache Size'])
        df = df[(df['Cache Size'] == cache_size)]

        df = df.sort_values(by=['degree'], ascending=[False])

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        x_data = list(map(lambda v: str(int(v)), df['degree']))
        ax1.plot(x_data, [(100 - optlocal) - (100 - optsplice) for optlocal, optsplice in zip(df['opt_local_result'],
                                                                                       df['opt_splice_result'])],
                                                                                       label="OptLocal-OptSplice",
                                                                                             marker="o")
        ax2.plot(x_data, df[rho_type], label=r"$\rho$", marker="d", color='red')

        xy_label_font_size = 28
        ax1.xaxis.set_tick_params(labelsize=xy_label_font_size)
        ax2.xaxis.set_tick_params(labelsize=xy_label_font_size)
        ax1.yaxis.set_tick_params(labelsize=xy_label_font_size)
        ax2.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax1.set_xlabel("degree", fontsize=xy_label_font_size)
        ax1.set_ylabel("Cache Miss Diff (%)", fontsize=xy_label_font_size)
        ax2.set_ylabel(r'$\rho$', fontsize=xy_label_font_size)

        # ax2.set_ylim([1, 110])
        # ax1.set_ylim([0, 30])
        # ax2.set_yscale('log')

        ax2.set_ylim([1, 350])
        ax1.set_ylim([0, 100])
        ax2.set_yscale('log')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines = lines_1 + lines_2
        labels = labels_1 + labels_2

        ax1.legend(lines, labels, prop=dict(size=16))
        ax1.grid(True)
        fig.tight_layout()


        if rho_type == 'new_rho':
            path_to_save = 'result/rho/sum60_70/Figures/' + csv_path.split('/')[-1].replace(
                '.csv', '_new_rho_diff_{0}.png'.format(int(cache_size)))
        else:
            path_to_save = 'result/rho/sum60_70/Figures/' + csv_path.split('/')[-1].replace(
                '.csv', '_diff_{0}.png'.format(int(cache_size)))

        print(path_to_save)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)

    @staticmethod
    def calculate_rho_for_weight_distribution(json_path):
        with open(json_path, 'r') as f:
            prefix2weight = json.load(f)

        node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(prefix2weight.keys())
        rule_to_vertex = {v: k for k, v in vertex_to_rule.items()}
        vtx2weight = {rule_to_vertex[p]: int(w) for p, w in prefix2weight.items()}
        return SyntheticRho.calc_rho(node_data_dict, vtx2weight)

    @staticmethod
    def plot_real_traces_rho():
        data = {
            "Stanford\nRandom": SyntheticRho.calculate_rho_for_weight_distribution(
                'traces/zipf_trace_1_0_prefix2weight.json'),
            "Standford\nby height": SyntheticRho.calculate_rho_for_weight_distribution(
                'traces/prefix2weight_sum60_70sorted_by_node_depth.json'),
            "Caida\nA": SyntheticRho.calculate_rho_for_weight_distribution('traces/caida_traceTCP_prefix_weight.json'),
            "Caida\nB": SyntheticRho.calculate_rho_for_weight_distribution('traces/caida_traceUDP_prefix_weight.json')}

        data = {k: np.around(v, 2) for k, v in sorted(data.items(), key=lambda val: val[1])}

        fig, ax = plt.subplots()
        ax.plot(list(data.keys()), list(data.values()), marker="d", color="red", markersize=14)

        xy_label_font_size = 28
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_ylabel(r'$\rho$', fontsize=xy_label_font_size)
        # ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
        ax.legend(prop=dict(size=16))
        ax.grid(True)

        # plt.show()
        path_to_save = 'result/Figures/new_rho/rho_real_traces.png'
        fig.tight_layout()
        print(path_to_save)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)

    @staticmethod
    def plot_by_rho(csv_dir):
        df = pd.DataFrame()
        for csv_file in filter(lambda d: 'csv' in d, os.listdir(csv_dir)):
            print(csv_file)
            curr_df = pd.read_csv(csv_dir + csv_file)
            curr_df['type'] = [csv_file.split('.')[0]] * curr_df.shape[0]
            df = pd.concat([df, curr_df])

        """
                ax.plot(list(map(str, true_df['Cache Size'])), list(map(lambda v: 100 - v, true_df['Hit Rate'])), marker="s",
                markersize=14, label="GreedySplice")
        ax.plot(list(map(str, false_df['Cache Size'])), list(map(lambda v: 100 - v, false_df['Hit Rate'])), marker="o",
                markersize=14, label="OptLocal")
        ax.plot(list(map(lambda x: str(int(x)), opt_df['Cache Size'])),
                list(map(lambda v: 100 - v, opt_df['Hit Rate'])), marker="P",
                markersize=14, label="OptLocal")
        """

        df = df[(df['Cache Size'] == 1024)].sort_values(['new_rho'], ascending=[True])
        df['rho_int'] = list(map(int, df['new_rho']))
        df = df.drop_duplicates(subset=['rho_int'])
        df = df[(df['Cache Size'] == 1024)].sort_values(['new_rho'], ascending=[True])
        fig, ax = plt.subplots()
        ax.plot(df['new_rho'], hit_to_miss(df['opt_local_result']), marker="o", markersize=14, label="OptLocal")
        ax.plot(df['new_rho'], hit_to_miss(df['opt_splice_result']), marker="P", markersize=14, label="OptSplice")
        # y_data = [(100 - x - (100 -y)) for x,y in zip(df['opt_local_result'], df['opt_splice_result'])]
        # ax.plot(df['new_rho'], y_data,
        #         marker="o", markersize=14, label="OptLocal-OptSplice")

        xy_label_font_size = 28
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_xlabel(r'$\rho$', fontsize=xy_label_font_size)
        ax.set_ylabel('Cache Miss (%)', fontsize=xy_label_font_size)
        # ax.set_ylabel('Cache Miss Diff (%)', fontsize=xy_label_font_size)

        ax.set_xscale('log', basex=2)
        # ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
        ax.legend(prop=dict(size=16))
        ax.grid(True)

        path_to_save = 'result/Figures/new_rho/xaxis_rho_cache_miss.png'

        fig.tight_layout()
        print(path_to_save)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)


def cache_miss_main():
    PlotResultTable.plot_range_result_table(
        "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/b1_t0/b1_t0.csv")
    PlotResultTable.plot_special_sort_result_table(
        "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/sorted_by_depth/sum60_70/sum60_70.csv")
    PlotResultTable.plot_special_sort_result_table(
        "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/result/UDP/UDP.csv")
    PlotResultTable.plot_special_sort_result_table(
        "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/result/TCP/TCP.csv")


def calculate_bar_data():
    hisogram_json_array = []
    base = "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/"
    # policy = base + "bar_weight_data/caida_traceUDP_policy.json"
    # prefix2weight = base + 'bar_weight_data/caida_traceUDP_prefix_weight.json'
    # height2weight_histogram_json_path = base + "bar_weight_data/UDP_height2weight_histogram.json"
    # # PlotResultTable.calculate_and_save_bar_data(policy, prefix2weight, height2weight_histogram_json_path)
    # hisogram_json_array.append(height2weight_histogram_json_path)
    #
    # policy = base + "bar_weight_data/caida_traceTCP_policy.json"
    prefix2weight = base + 'bar_weight_data/caida_traceTCP_prefix_weight.json'
    height2weight_histogram_json_path = base + "bar_weight_data/TCP_height2weight_histogram.json"
    # PlotResultTable.calculate_and_save_bar_data(policy, prefix2weight, height2weight_histogram_json_path)
    hisogram_json_array.append(height2weight_histogram_json_path)
    #
    # base = "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/"
    # policy = base + "bar_weight_data/prefix_only.json"
    # # prefix2weight = base + 'bar_weight_data/prefix2weight_sum60_70sorted_by_node_depth.json'
    # height2weight_histogram_json_path = base + "bar_weight_data/sum60_70sorted_by_node_depth_height2weight_histogram.json"
    # # PlotResultTable.calculate_and_save_bar_data(policy, prefix2weight, height2weight_histogram_json_path)
    # hisogram_json_array.append(height2weight_histogram_json_path)
    #
    prefix2weight = base + 'bar_weight_data/zipf_trace_1_0_prefix2weight.json'
    height2weight_histogram_json_path = base + "bar_weight_data/zipf_trace_1_0_prefix2weight_height2weight_histogram.json"
    # PlotResultTable.calculate_and_save_bar_data(policy, prefix2weight, height2weight_histogram_json_path)
    hisogram_json_array.append(height2weight_histogram_json_path)

    return hisogram_json_array


def bar_plot_main():
    base = "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/bar_plot/"
    path2save_arr = [
        # 'caida_traceUDP_prefix_weight.jpg',
        'caida_traceTCP_prefix_weight.jpg',
        # 'prefix2weight_sum60_70sorted_by_node_depth.jpg'
        'zipf_trace_1_0_prefix2weight.jpg'
    ]
    hisogram_json_array = calculate_bar_data()
    for hisogram_json, path2save in zip(hisogram_json_array, path2save_arr):
        PlotResultTable.plot_weight_bar(hisogram_json, base + path2save)


def parse_mrt():
    from mrtparse import Reader
    path = "../rib.20180205.1800"
    # with open(path, 'r') as f:
    for entry in Reader(path):
        print(entry.data.get('prefix', ''))
        # print(entry.data['prefix'])


def create_heatmap():
    base = "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/"
    policy = base + "bar_weight_data/caida_traceUDP_policy.json"
    prefix2weight = base + 'bar_weight_data/caida_traceUDP_prefix_weight.json'
    UDP_heatmap_data = base + "bar_weight_data/UDP_heatmap_data.npy"
    PlotResultTable.plot_heatmap(policy, prefix2weight, UDP_heatmap_data)

def join_csvs(dataset_random, dataset_ascending, dataset_descending):
    df_random = pd.read_csv(dataset_random)
    df_ascending = pd.read_csv(dataset_ascending)
    df_descending = pd.read_csv(dataset_descending)

    df_random = df_random[(df_random['Cache Size'] == 1024)]
    df_ascending = df_ascending[(df_ascending['Cache Size'] == 1024)]
    df_descending = df_descending[(df_descending['Cache Size'] == 1024)]

    df_random['type'] = ['random']*df_random.shape[0]
    df_ascending['type'] = ['ascending']*df_ascending.shape[0]
    df_descending['type'] = ['descending']*df_descending.shape[0]

    final_df = pd.concat([df_descending, df_random, df_ascending])
    final_df.to_csv('result/rho/result_summary.csv')
    print("s")


def main():
    # RunCheck.test_random_OptLPM()
    # RunCheck.running_example()
    # parse_mrt()
    # RunCheck.compare_greedy_opt_solution()
    # SyntheticRho.generate_rho_experiment('result/rho')
    # SyntheticRho.plot_csv_rho_diff('result/rho/dataset_sorted_ascending_depth.csv',rho_type='new_rho')
    # SyntheticRho.plot_csv_rho_diff('result/rho/dataset_sorted_descending_depth.csv',rho_type='new_rho')
    # SyntheticRho.plot_csv_rho_diff('result/rho/dataset_random.csv',rho_type='new_rho')
    # for cache_size in [1024]:#, 512, 64]:
    #     SyntheticRho.plot_csv_rho('result/rho/dataset_sorted_ascending_depth.csv',rho_type="new_rho", cache_size=cache_size)
    #     SyntheticRho.plot_csv_rho('result/rho/dataset_sorted_descending_depth.csv',rho_type="new_rho",cache_size=cache_size)
    #     SyntheticRho.plot_csv_rho('result/rho/dataset_random.csv',rho_type="new_rho",cache_size=cache_size)

    # SyntheticRho.create_zipf(6000, 'result/rho')




if __name__ == "__main__":
    main()
