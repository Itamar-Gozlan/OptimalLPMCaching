import json
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

ROOT = 0
ROOT_PREFIX = '0.0.0.0/0'


class RunCheck:
    @staticmethod
    def get_random_policy_and_weight():
        policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(15)]
        policy_weight = {k.strip(): np.random.randint(100) for k in policy}
        print("policy = {0}".format(policy))
        print("policy_weight = {0}".format(policy_weight))

        if '0.0.0.0/0' not in policy:
            policy.append("0.0.0.0/0")
        policy_weight['0.0.0.0/0'] = 0

        return policy, policy_weight

    @staticmethod
    def validate_online_optimal_lpm():
        policy, policy_weight, packet_trace = RunCheck.get_random_policy_and_weight()
        cache_size = 3
        dependency_splice = False

        # Utils.draw_tree(offline.policy_tree, {v : v for v in offline.policy_tree.nodes})
        # plt.show()
        weights_up_to_i = {}
        for i in range(len(packet_trace)):
            offline = HeuristicLPMCache(cache_size, policy, dependency_splice)
            online = OnlineOptimalLPMCache(cache_size, policy, dependency_splice)
            weights_up_to_i = {}
            for packet in packet_trace[:i + 1]:
                if packet not in online.cache:
                    weights_up_to_i[packet] = 1 + weights_up_to_i.get(packet, 0)
                    online.cache_miss(packet)
            get_offline_cache = offline.get_cache(weights_up_to_i)
            # print(online.rule_counter == {online.rule_to_vertex[r]:v for r,v in weights_up_to_i.items()})
            print("Round {0}, Test: {1}".format(i, online.cache == get_offline_cache))

        # get_offline_cache = offline.get_cache(policy_weight)
        # for packet in packet_trace:
        #     if packet not in online.cache:
        #         online.cache_miss(packet)
        #
        # print(get_offline_cache == online.cache)
        # print("s")

    @staticmethod
    def test_online_tree_cache():
        with open('Caida/6000rules_small/caida_traceTCP_policy.json', 'r') as f:
            policy = json.load(f)
            cache_size = 256
            OTC = OnlineTreeCache(policy, cache_size)
            print("s")

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
    def test_random_OptLPM():
        # policy, policy_weight = RunCheck.get_random_policy_and_weight()
        policy_weight = {"0.0.0.0/0": 0,
                         # "10.0.0.0/18": 12,  # R1
                         # "10.0.0.0/20": 17,  # R2
                         # "10.0.64.0/18": 25,  # R3
                         # "10.0.0.0/8": 25,  # R3
                         "10.0.0.0/8": 25,  # R3
                         # "10.0.64.0/20": 3,  # R4
                         "10.0.64.0/22": 5,  # R5
                         # "10.0.80.0/22": 10,  # R6
                         "10.0.80.0/24": 15,  # R7
                         "10.0.56.0/22" : 15 # R8
                         }

        prefix_to_rule_name = {"0.0.0.0/0": "R0",
                               "10.0.0.0/18": "R1",
                               "10.0.0.0/20": "R2",
                               "10.0.0.0/8": "R3",
                               "10.0.64.0/20": "R4",
                               "10.0.64.0/22": "R5",
                               "10.0.80.0/22": "R6",
                               "10.0.80.0/24": "R7",
                               "10.0.56.0/22" : "R8"
                               }

        policy_weight = {"0.0.0.0/0": 0,
                         "10.0.0.0/8": 1000,  # R1
                         "10.0.0.0/16": 10,  # R1
                         "10.0.1.0/24": 5,  # R1
                         "10.0.2.0/24": 20,  # R1
                         "10.0.3.0/24": 15,  # R1
                         }

        policy = list(policy_weight.keys())
        cache_size = 4
        algorithm = OptimalLPMCache(policy, policy_weight, cache_size)
        algorithm.get_optimal_cache()

        color_map = []
        for vtx in algorithm.policy_tree.nodes:
            if vtx in algorithm.S[ROOT][0][cache_size]:
                color_map.append('red')
                continue
            if vtx in algorithm.gtc_nodes:
                color_map.append('black')
                continue
            else:
                color_map.append('green')


        labels = {v: "R_{0}, {1}".format(idx, policy_weight[algorithm.vertex_to_rule[v]])
                  for idx, v in enumerate(algorithm.policy_tree.nodes)}



        # nx.draw(algorithm.policy_tree, labels=labels)
        Utils.draw_tree(algorithm.policy_tree, labels, color_map=color_map)
        plt.show()


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


class PlotResultTable:
    @staticmethod
    def plot_true_false_df(true_df, false_df, path_to_save, ylim=[0, 80]):
        fig, ax = plt.subplots()
        ax.plot(list(map(str, true_df['Cache Size'])), list(map(lambda v: 100 - v, true_df['Hit Rate'])), marker="s",
                markersize=14, label="OptSplice'")
        ax.plot(list(map(str, false_df['Cache Size'])), list(map(lambda v: 100 - v, false_df['Hit Rate'])), marker="o",
                markersize=14, label="OptLocal")
        xy_label_font_size = 28
        ax.xaxis.set_tick_params(labelsize=xy_label_font_size)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_ylabel('Cache Miss (%)', fontsize=xy_label_font_size)
        ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
        ax.set_ylim(ylim)
        ax.legend(prop=dict(size=24))
        ax.grid(True)

        fig.tight_layout()
        print(path_to_save)
        h = 4
        fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        fig.savefig(path_to_save, dpi=300)

    @staticmethod
    def plot_range_result_table(csv_path):
        df = pd.read_csv(csv_path)
        cache_size_array = sorted(list(df['Cache Size'].drop_duplicates()))
        for index, row in df[['Bottom', 'Top']].drop_duplicates().iterrows():
            top = row['Top']
            bottom = row['Bottom']
            res_df = df[(df['Bottom'] == bottom) &
                        (df['Top'] == top)]
            true_df = res_df[(res_df['Splice'] == True)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
            false_df = res_df[(res_df['Splice'] == False)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')

        PlotResultTable.plot_true_false_df(true_df, false_df, csv_path.replace('csv', 'jpg'))

        print("s")

    @staticmethod
    def plot_special_sort_result_table(csv_path):
        df = pd.read_csv(csv_path)

        true_df = df[(df['Splice'] == True)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
        false_df = df[(df['Splice'] == False)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')

        PlotResultTable.plot_true_false_df(true_df, false_df, csv_path.replace('csv', 'jpg'))

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
    def plot_bar(x_data, y_data, path_to_save):

        fig, ax = plt.subplots()
        ax.bar(x_data, y_data)
        xy_label_font_size = 25
        ax.xaxis.set_tick_params(labelsize=xy_label_font_size)
        # ax.set_yticks([1, 2, 3, 4, 5])
        ax.yaxis.set_tick_params(labelsize=xy_label_font_size)

        ax.set_ylabel('Average subtree size', fontsize=xy_label_font_size)
        ax.set_xlabel("Height", fontsize=xy_label_font_size)
        ax.set_yscale('log', basey=2)

        ax.set_ylim([1, 300])
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
    def plot_heatmap(policy_json, prefix_weight_json, path_to_save):
        with open(policy_json, 'r') as f:
            policy = json.load(f)
        node_data_dict, vertex_to_rule = NodeData.construct_node_data_dict(policy)
        rule_to_vertex = {v: k for k, v in vertex_to_rule.items()}

        with open(prefix_weight_json, 'r') as f:
            prefix_weight = json.load(f)

        node_height = {v: node_data_dict[v].subtree_depth for v in node_data_dict}
        bin = 0
        # log2_y = [0.45, 0.9, 1]
        # log2_y = np.linspace(0,1,4)
        # log2_y = [0.5, 0.75, 0.85, 1]
        # log2_y = [0.5, 0.75,. ]
        # z = [1, 0.5, 0.25, 0.125]
        # log2_y = [0.5, 0.75, 0.875, 0.9375]
        log2_y = [0.001, 1]  # 0.75, 0.875, 0.9375]#, 0.96875, 0.984375, 0.9921875]
        log2_y = []
        # log2_y = [1-(1/2)**i for i in range(8)]
        sum_total = sum(prefix_weight.values())

        n_rows = len(log2_y)
        n_cols = len(set(node_height.values()))
        heatmap_data = np.zeros((n_rows, n_cols))
        for prefix, weight in sorted(prefix_weight.items(), key=lambda key: key[1]):
            height = node_height.get(rule_to_vertex[prefix], 0)
            traffic_p = weight / sum_total
            while traffic_p > log2_y[bin]:
                bin += 1
            # found rule with weight in bin
            heatmap_data[bin][height - 1] += 1

        np.save(path_to_save, heatmap_data)
        # heatmap_data = np.load(path_to_save)

        ax = sns.heatmap(heatmap_data)
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

        # fig.tight_layout()
        # print(path_to_save)
        # h = 4
        # fig.set_size_inches(h * (1 + 5 ** 0.5) / 2, h * 1.1)
        # fig.savefig(path_to_save, dpi=300)


class MakeRunScript:
    @staticmethod
    def make_run_script_range():
        arr = ["zipf_trace_10_50_packet_array.json",
               "zipf_trace_10_50_prefix2weight.json",
               "zipf_trace_1_0_packet_array.json",
               "zipf_trace_1_0_prefix2weight.json",
               "zipf_trace_20_50_packet_array.json",
               "zipf_trace_20_50_prefix2weight.json",
               "zipf_trace_2_50_packet_array.json",
               "zipf_trace_2_50_prefix2weight.json",
               "zipf_trace_30_50_packet_array.json",
               "zipf_trace_30_50_prefix2weight.json",
               "zipf_trace_40_50_packet_array.json",
               "zipf_trace_40_50_prefix2weight.json"]

        trace_prefix_array = ['zipf_trace_10_50',
                              'zipf_trace_1_0',
                              'zipf_trace_20_50',
                              'zipf_trace_2_50',
                              'zipf_trace_30_50',
                              'zipf_trace_40_50']

        suffix = ['_prefix2weight.json, _packet_array.json']
        opt_array = ["True", "False"]
        cache_size_array = [64, 128, 256, 512]

        base_path = "run_sc/"
        count = 0
        cmd_array = []
        for trace_prefix in trace_prefix_array:
            for opt in opt_array:
                for cache_size in cache_size_array:
                    if count % 6 == 0:
                        if len(cmd_array) > 0:
                            with open(base_path + "run_sc{0}.sh".format(int(count / 6) - 1), 'w') as f:
                                f.writelines(sorted(cmd_array, key=lambda cmd: int(cmd.split(" ")[4])))
                            cmd_array = []
                    cmd = "python ../simulator_main.py ../traces/{0}_prefix2weight.json ../traces/{0}_packet_array.json {1} {2} > offline_{0}_{1}_{2}.out\n".format(
                        trace_prefix, cache_size, opt)
                    cmd_array.append(cmd)
                    count += 1

        with open(base_path + "run_sc{0}.sh".format(int(count / 6) - 1), 'w') as f:
            f.writelines(sorted(cmd_array, key=lambda cmd: int(cmd.split(" ")[4])))

        print(" =================== ")
        for i in range(int(count / 6)):
            print("chmod +x run_sc{0}.sh".format(i))

    @staticmethod
    def create_caida_offline_run():
        """
        policy = sys.argv[1]
        packet_trace_json_path = sys.argv[2]
        cache_size = int(sys.argv[3])
        dependency_splice = eval(sys.argv[4])
        """

        base = "/home/user46/OptimalLPMCaching/Caida/6000rules/"
        base = "/home/user46/OptimalLPMCaching/Caida/caida_final/"
        arr = [("caida_traceTCP_prefix_weight.json", "caida_traceTCP_packet_array.json"),
               ("caida_traceUDP_prefix_weight.json", "caida_traceUDP_packet_array.json")
               ]
        opt_array = ["True", "False"]
        cache_size_array = [64, 128, 256, 512, 1024]
        cmd_array = []
        for pfx_weight, packet_trace in arr:
            for opt in opt_array:
                for cache_size in cache_size_array:
                    exp_name = "_".join(packet_trace.split('_')[-3:-2])
                    cmd = "python ../simulator_main.py {5}{0} {5}{1} {2} {3} > {5}result/caida_{4}_{2}_{3}.out \n".format(
                        pfx_weight, packet_trace, cache_size, opt, exp_name, base)
                    cmd_array.append(cmd)

        sorted_by_processing_time = sorted(cmd_array, key=lambda cmd: int(cmd.split(" ")[4]))
        n_files = 5
        array_to_file = [[] for i in range(n_files)]
        i = 0
        while sorted_by_processing_time:
            print("i % n_files : {0}".format(i % n_files))
            print("len(array_to_file[i % n_files]) : {0}".format(len(array_to_file[i % n_files])))
            cmd = sorted_by_processing_time.pop(0)
            array_to_file[i % n_files].append(cmd)
            print("len(array_to_file[i % n_files]) : {0}".format(len(array_to_file[i % n_files])))
            i += 1

        for i in range(n_files):
            with open('run_sc/run_sc_C{0}.sh'.format(i), 'w') as f:
                f.writelines(array_to_file[i])

        print(" =================== ")
        for i in range(n_files):
            print("chmod +x run_sc_C{0}.sh".format(i))

        print(" =================== ")
        for i in range(n_files):
            print("nohup ./run_sc_C{0}.sh > run_sc_C{0}.out &".format(i))

    @staticmethod
    def make_run_script_special_sort():
        arr = ["zipf_weight_30M_sum60_70sorted_by_subtree_size_dvd_n_children.json_packet_array.json",
               "zipf_weight_30M_sum60_70sorted_by_subtree_size_dvd_n_children.json_prefix2weight.json",
               "zipf_weight_30M_sum60_70_sorted_by_subtree_size.json_packet_array.json",
               "zipf_weight_30M_sum60_70_sorted_by_subtree_size.json_prefix2weight.json",
               "zipf_weight_30M_sum60_90sorted_by_subtree_size_dvd_n_children.json_packet_array.json",
               "zipf_weight_30M_sum60_90sorted_by_subtree_size_dvd_n_children.json_prefix2weight.json",
               "zipf_weight_30M_sum60_90_sorted_by_subtree_size.json_packet_array.json",
               "zipf_weight_30M_sum60_90_sorted_by_subtree_size.json_prefix2weight.json"]

        format_packet_array_name = lambda p: ('_'.join(p.split('_')[-2:]) + '_' + "_".join(p.split('_')[3:-2])).replace(
            '.json', '') + '.json'
        format_prefix2weight = lambda p: (p.split('_')[-1] + "_" + "_".join(p.split('_')[3:-1])).replace('.json',
                                                                                                         '') + '.json'

        for n in arr:
            if 'prefix2weight' in n:
                dst_name = format_prefix2weight(n)
            else:
                dst_name = format_packet_array_name(n)
            print("mv {0} {1}".format(n, dst_name))

        file_name_array = ['sum60_70_sorted_by_subtree_size.json',
                           'sum60_70sorted_by_subtree_size_dvd_n_children.json',
                           'sum60_90_sorted_by_subtree_size.json',
                           'sum60_90sorted_by_subtree_size_dvd_n_children.json']
        packet_array_prefix = "packet_array_"
        pfx2weight_prefix = "prefix2weight_"

        cache_size_array = [64, 128, 256, 512]
        opt_array = [True, False]
        cmd_array = []
        for file_name in file_name_array:
            for opt in opt_array:
                for cache_size in cache_size_array:
                    cmd = "python ../simulator_main.py ../traces/prefix2weight_{0} ../traces/packet_array_{0} {1} {2} > offline_{1}_{2}_{3}.out\n".format(
                        file_name, cache_size, opt, file_name.replace('.json', ''))
                    cmd_array.append(cmd)
                    print(cmd)

        sorted_by_processing_time = sorted(cmd_array, key=lambda cmd: int(cmd.split(" ")[4]))
        n_files = 8
        array_to_file = [[] for i in range(n_files)]
        i = 0
        while sorted_by_processing_time:
            print("i % n_files : {0}".format(i % n_files))
            print("len(array_to_file[i % n_files]) : {0}".format(len(array_to_file[i % n_files])))
            cmd = sorted_by_processing_time.pop(0)
            array_to_file[i % n_files].append(cmd)
            print("len(array_to_file[i % n_files]) : {0}".format(len(array_to_file[i % n_files])))
            i += 1

        for i in range(n_files):
            with open('run_sc/run_sc_X{0}.sh'.format(i), 'w') as f:
                f.writelines(array_to_file[i])

        print(" =================== ")
        for i in range(n_files):
            print("chmod +x run_sc_X{0}.sh".format(i))

        print(" =================== ")
        for i in range(n_files):
            print("nohup ./run_sc_X{0}.sh > run_sc_X{0}.out &".format(i))

        print("s")

    @staticmethod
    def construct_cmd_array_OTC(trace_name_array, base, out_file_to_cmd, policy_json_path):
        cmd_array = []
        simulator_main = '/home/user46/OptimalLPMCaching/simulator_main.py'
        result_dir = '/home/user46/OptimalLPMCaching/OTC_result/'
        cache_size_array = [64, 128, 256, 512]
        for trace_name in trace_name_array:
            for cache_size in cache_size_array:
                out_file = result_dir + trace_name.replace('packet_array', '').replace('.json', '') + '{0}_{1}'.format(
                    trace_name.split('.')[-1], cache_size) + '.out'
                # args: policy_json_path packet_trace_json_path cache_size
                cmd = 'python ' + simulator_main + " {1} {0}{2} {3} > {4}\n".format(base,
                                                                                    policy_json_path,
                                                                                    trace_name,
                                                                                    cache_size,
                                                                                    out_file)
                out_file_to_cmd[out_file] = cmd
                cmd_array.append(cmd)
        return cmd_array

    @staticmethod
    def construct_cmd_array_online_experiment(trace_name_array, base, out_file_to_cmd, policy_json_path):
        cmd_array = []
        simulator_main = '/home/user46/OptimalLPMCaching/simulator_main.py'
        result_dir = '/home/user46/OptimalLPMCaching/OptLPM_online/'
        cache_size_array = [64, 128, 256, 512]
        epoch_array = [1.0]  # , 0.5, 1.0]
        opt_array = [True, False]
        for trace_name in trace_name_array:
            for cache_size in cache_size_array:
                for epoch in epoch_array:
                    for opt in opt_array:
                        # policy = sys.argv[1]
                        # packet_trace_json_path = sys.argv[2]
                        # cache_size = int(sys.argv[3])
                        # dependency_splice = eval(sys.argv[4])
                        # epoch = int(sys.argv[5])
                        out_file = (result_dir + trace_name + '_{0}_{1}_{2}'.format(cache_size, opt, epoch)).replace(
                            '.json', '') + '.out'
                        # args: policy_json_path packet_trace_json_path cache_size
                        cmd = 'python ' + simulator_main + " {1} {0}{2} {3} {4} {5} > {6}\n".format(base,
                                                                                                    policy_json_path,
                                                                                                    trace_name,
                                                                                                    cache_size,
                                                                                                    opt,
                                                                                                    epoch,
                                                                                                    out_file)
                        out_file_to_cmd[out_file] = cmd
                        cmd_array.append(cmd)
        return cmd_array

    @staticmethod
    def cmd_array_to_run_files(n_files, cmd_array, lambda_key_sort):
        sorted_by_processing_time = sorted(cmd_array, key=lambda_key_sort)
        array_to_file = [[] for i in range(n_files)]
        i = 0
        while sorted_by_processing_time:
            # print("i % n_files : {0}".format(i % n_files))
            # print("len(array_to_file[i % n_files]) : {0}".format(len(array_to_file[i % n_files])))
            cmd = sorted_by_processing_time.pop(0)
            array_to_file[i % n_files].append(cmd)
            # print("len(array_to_file[i % n_files]) : {0}".format(len(array_to_file[i % n_files])))
            i += 1

        for i in range(n_files):
            with open('run_sc/run_sc_X{0}.sh'.format(i), 'w') as f:
                f.writelines(array_to_file[i])

        print(" =================== ")
        for i in range(n_files):
            print("chmod +x run_sc_X{0}.sh".format(i))

        print(" =================== ")
        for i in range(n_files):
            print("nohup ./run_sc_X{0}.sh > run_sc_X{0}.out &".format(i))

    @staticmethod
    def make_run_sc():
        out_file_to_cmd = {}
        cmd_array = []
        construct_cmd_array = MakeRunScript.construct_cmd_array_online_experiment
        cmd_to_cache_size = lambda cmd_str: (float(cmd_str.split(' ')[-3]), int(cmd_str.split(' ')[-5]))

        # construct_cmd_array = MakeRunScript.construct_cmd_array_OTC
        # cmd_to_cache_size = lambda cmd_str: int(cmd_str.split(' ')[-3])

        policy_json_path = "/home/user46/OptimalLPMCaching/Zipf/prefix_only.json"

        base = "/home/user46/OptimalLPMCaching/traces/sum60_90/"
        trace_name_array = ["zipf_trace_10_50_packet_array.json",
                            "zipf_trace_1_0_packet_array.json",
                            "zipf_trace_20_50_packet_array.json",
                            "zipf_trace_2_50_packet_array.json",
                            "zipf_trace_30_50_packet_array.json",
                            "zipf_trace_40_50_packet_array.json"]
        cmd_array += construct_cmd_array(trace_name_array, base, out_file_to_cmd, policy_json_path)

        policy_json_path = "caida_traceUDP_policy.json"
        base = "/home/user46/OptimalLPMCaching/Caida/6000rules/"
        trace_name_array = ["caida_traceUDP_packet_array.json"]
        cmd_array += construct_cmd_array(trace_name_array, base, out_file_to_cmd, base + policy_json_path)

        policy_json_path = "caida_traceTCP_policy.json"
        base = "/home/user46/OptimalLPMCaching/Caida/6000rules/"
        trace_name_array = ["caida_traceTCP_packet_array.json"]
        cmd_array += construct_cmd_array(trace_name_array, base, out_file_to_cmd, base + policy_json_path)

        # policy_json_path = "/home/user46/OptimalLPMCaching/Zipf/prefix_only.json"
        # base = "/home/user46/OptimalLPMCaching/traces/sum60_70/"
        # trace_name_array = ["zipf_trace_10_50_packet_array.json"
        #                     "zipf_trace_1_0_packet_array.json",
        #                     "zipf_trace_20_50_packet_array.json",
        #                     "zipf_trace_2_50_packet_array.json",
        #                     "zipf_trace_30_50_packet_array.json",
        #                     "zipf_trace_40_50_packet_array.json"]
        # cmd_array += construct_cmd_array(trace_name_array, base, out_file_to_cmd, policy_json_path)

        # base = "/home/user46/OptimalLPMCaching/traces/special_sort/"
        # trace_name_array = ["packet_array_sum60_70sorted_by_node_depth.json",
        #                     "packet_array_sum60_70sorted_by_subtree_size_dvd_n_children.json",
        #                     "packet_array_sum60_70_sorted_by_subtree_size.json",
        #                     "packet_array_sum60_90sorted_by_node_depth.json",
        #                     "packet_array_sum60_90sorted_by_subtree_size_dvd_n_children.json",
        #                     "packet_array_sum60_90_sorted_by_subtree_size.json"]
        # cmd_array += construct_cmd_array(trace_name_array, base, out_file_to_cmd, policy_json_path)

        print(len(cmd_array))
        print(cmd_array[0])
        MakeRunScript.cmd_array_to_run_files(5, cmd_array, lambda_key_sort=cmd_to_cache_size)


def place_holder():
    pass
    # PlotResultTable.plot_range_result_table("Figures/2707_results/offline_sum60_90/offline_sum60_90_result.csv")

    # format_result_into_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_70_sorted_by_subtree_size")
    # format_result_into_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_90_sorted_by_subtree_size")
    # format_result_into_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_70sorted_by_subtree_size_dvd_n_children")
    # format_result_into_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_90sorted_by_subtree_size_dvd_n_children")

    # plot_special_sort_result_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_70_sorted_by_subtree_size/sum60_70_sorted_by_subtree_size.csv")
    # plot_special_sort_result_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_90_sorted_by_subtree_size/sum60_90_sorted_by_subtree_size.csv")
    # plot_special_sort_result_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_70sorted_by_subtree_size_dvd_n_children/sum60_70sorted_by_subtree_size_dvd_n_children.csv")
    # plot_special_sort_result_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/special_sort/sum60_90sorted_by_subtree_size_dvd_n_children/sum60_90sorted_by_subtree_size_dvd_n_children.csv")

    # MakeRunScript.make_run_script_special_sort()

    # make_run_script()

    # create_caida_offline_run()

    # format_result_into_table("/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/6000rules/result")
    # plot_special_sort_result_table(
    #     "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/6000rules/result/result_dir/result_udp.csv")

    # format_result_into_table("/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/caida_final/result/UDP")
    # plot_special_sort_result_table(
    # "/home/itamar/PycharmProjects/OptimalLPMCaching/Caida/6000rules/result/result_dir/result_udp.csv")

    # for X in ["sorted_by_subtree_size_dvd_n_children", "sorted_by_subtree_size", "sorted_by_depth"]:
    #     for Y in [70, 90]:
    #         ResultToTable.format_special_sort(
    #             "/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/{0}/sum60_{1}".format(X, Y))
    #         PlotResultTable.plot_special_sort_result_table(
    #             "/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/{0}/sum60_{1}/sum60_{1}.csv".format(X, Y))

    # PlotResultTable.plot_caida_result_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/6000rules/result_dir/result_tcp.csv")
    # PlotResultTable.plot_caida_result_table("/home/itamar/PycharmProjects/OptimalLPMCaching/run_sc/6000rules/result_dir/result_udp.csv")
    # RunCheck.validate_online_optimal_lpm()
    # MakeRunScript.make_run_sc()

    # ResultToTable.format_result_into_table('C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/sorted_by_depth/sum60_70')

    # ResultToTable.format_result_into_table('C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/result/UDP')
    # ResultToTable.format_result_into_table('C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/result/TCP')
    # bar_plot_main() # plot_weight_bar

    """

    base = "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/"
    policy = base + "bar_weight_data/caida_traceUDP_policy.json"
    json_path = base + 'subtree_size/caida_traceUDP.json'
    # PlotResultTable.calculate_and_save_bin_bar_data_subtree_size(policy, json_path)
    PlotResultTable.plot_subtree_bar(json_path, json_path.replace('json', 'jpg'))

    policy = base + "bar_weight_data/caida_traceTCP_policy.json"
    json_path = base + 'subtree_size/caida_traceTCP.json'
    # PlotResultTable.calculate_and_save_bin_bar_data_subtree_size(policy, json_path)
    PlotResultTable.plot_subtree_bar(json_path, json_path.replace('json', 'jpg'))

    # policy = base + "bar_weight_data/prefix_only.json"
    # json_path = base + 'subtree_size/stanford_backbone.json'
    # PlotResultTable.calculate_and_save_bin_bar_data_subtree_size(policy, json_path)
    # PlotResultTable.plot_subtree_bar(json_path, json_path.replace('json', 'jpg'))
    """

    cache_miss_main()


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


def create_heatmap():
    base = "C:/Users/Hadar Matlaw/Desktop/Itamar/OptimalLPMCaching/last_min_additions/"
    policy = base + "bar_weight_data/caida_traceUDP_policy.json"
    prefix2weight = base + 'bar_weight_data/caida_traceUDP_prefix_weight.json'
    UDP_heatmap_data = base + "bar_weight_data/UDP_heatmap_data.npy"
    PlotResultTable.plot_heatmap(policy, prefix2weight, UDP_heatmap_data)


def main():
    RunCheck.test_random_OptLPM()


if __name__ == "__main__":
    main()
