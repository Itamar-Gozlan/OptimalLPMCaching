import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from Zipf import produce_packet_array_of_prefix_weight
import Utils
from Algorithm import *
import time
import pandas as df

ROOT = 0
ROOT_PREFIX = '0.0.0.0/0'


class RunCheck:
    @staticmethod
    def get_random_policy_and_weight():
        # policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(15)]
        # policy_weight = {k.strip(): np.random.randint(100) for k in policy}
        # print("policy = {0}".format(policy))
        # print("policy_weight = {0}".format(policy_weight))
        policy = ['237.200.0.0/15', '107.236.0.0/14', '239.128.0.0/10', '59.168.0.0/14', '89.0.0.0/10',
                  '199.156.64.0/19', '145.0.0.0/8', '145.40.0.0/13', '160.0.0.0/5', '192.0.0.0/2', '134.164.0.0/16',
                  '253.221.140.172/32', '51.0.0.0/8', '0.64.0.0/10', '158.7.193.96/29']
        policy_weight = {'237.200.0.0/15': 33, '107.236.0.0/14': 80, '239.128.0.0/10': 9, '59.168.0.0/14': 0,
                         '89.0.0.0/10': 63, '199.156.64.0/19': 62, '145.0.0.0/8': 58, '145.40.0.0/13': 92,
                         '160.0.0.0/5': 83, '192.0.0.0/2': 45, '134.164.0.0/16': 28, '253.221.140.172/32': 9,
                         '51.0.0.0/8': 46, '0.64.0.0/10': 20, '158.7.193.96/29': 47}

        if '0.0.0.0/0' not in policy:
            policy.append("0.0.0.0/0")
        policy_weight['0.0.0.0/0'] = 0
        # packet_trace = produce_packet_array_of_prefix_weight(policy_weight)
        packet_trace = ['51.0.0.0/8', '134.164.0.0/16', '160.0.0.0/5', '145.40.0.0/13', '134.164.0.0/16',
                        '253.221.140.172/32', '145.40.0.0/13', '192.0.0.0/2', '145.0.0.0/8', '145.40.0.0/13']
        return policy, policy_weight, packet_trace

    @staticmethod
    def validate_online_optimal_lpm():
        policy, policy_weight, packet_trace = RunCheck.get_random_policy_and_weight()
        cache_size = 3
        dependency_splice = False

        # Utils.draw_tree(offline.policy_tree, {v : v for v in offline.policy_tree.nodes})
        # plt.show()
        weights_up_to_i = {}
        for i in range(len(packet_trace)):
            offline = OptimalLPMCache(cache_size, policy, dependency_splice)
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
        with open('Caida/6000rules/caida_traceTCP_policy.json', 'r') as f:
            policy = json.load(f)
        # policy = {k: v.strip() for k,v in policy.items()}
        # policy = open('../Zipf/prefix_only.txt', 'r').read().splitlines()
        # with open('../Zipf/sorted_prefix_with_weights.json', 'r') as f:
        #     policy_weight = json.load(f)
        res = []
        # print(policy)
        cache_size = 256
        # policy = [Utils.binary_lpm_to_str(s) for s in Utils.compute_random_policy(10000)]
        OptLPMAlg = OptimalLPMCache(cache_size, policy, dependency_splice=True)

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
                else:
                    print("Missing: {0}".format(dirpath))
                    missing_result_array.append(dirpath)

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

            fig, ax = plt.subplots()
            ax.plot(list(map(str, true_df['Cache Size'])), true_df['Hit Rate'], marker="d", label="Dependency Splice")
            ax.plot(list(map(str, false_df['Cache Size'])), false_df['Hit Rate'], marker="o", label="Without Splice")
            # if bottom < top:
            #     ax.set_title("Heaviest Nodes: {0} <= degree <= {1}".format(bottom, top))
            # else:
            #     ax.set_title("Random Zipf".format(bottom, top))
            xy_label_font_size = 28
            # ax.set_yticklabels(fontsize=24)
            # ax.set_xticklabels(fontsize=24)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)

            ax.set_ylabel('Cache Hit (%)', fontsize=xy_label_font_size)
            ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
            ax.set_ylim([10, 100])
            ax.legend()

            experiment_name = "run_sc_local/2707_sum60_90_result.csv".split('/')[-1].replace('_result.csv', '')
            # path_to_save = 'Figures/{0}'.format(experiment_name)
            # if not os.path.exists(path_to_save):
            #     os.makedirs(path_to_save)
            filename = csv_path.split('/')[-1]
            path_to_save = csv_path.replace(filename, '') + '/b{0}_t{1}.jpg'.format(bottom, top)
            fig.tight_layout()
            print(path_to_save)
            fig.savefig(path_to_save, dpi=300)

        print("s")

    @staticmethod
    def plot_special_sort_result_table(csv_path):
        df = pd.read_csv(csv_path)

        true_df = df[(df['Splice'] == True)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
        false_df = df[(df['Splice'] == False)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')

        fig, ax = plt.subplots()
        ax.plot(list(map(str, true_df['Cache Size'])), true_df['Hit Rate'], marker="d", label="Dependency Splice")
        ax.plot(list(map(str, false_df['Cache Size'])), false_df['Hit Rate'], marker="o", label="Without Splice")

        xy_label_font_size = 28
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)

        ax.set_ylabel('Cache Hit (%)', fontsize=xy_label_font_size)
        ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
        ax.set_ylim([0, 100])
        ax.legend()
        fig.tight_layout()
        print(csv_path.replace('csv', 'jpg'))
        sort_type = csv_path.split('/')[-3]
        csv_dir_split, csv_name = csv_path.split('/')[:-1], csv_path.split('/')[-1]
        path_to_save = "/".join(csv_dir_split + [csv_name.replace('.csv', '_{0}.jpg'.format(sort_type))])
        print(path_to_save)
        fig.savefig(path_to_save, dpi=300)

        print("s")

    @staticmethod
    def plot_caida_result_table(csv_path):
        df = pd.read_csv(csv_path)

        true_df = df[(df['Splice'] == True)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')
        false_df = df[(df['Splice'] == False)][['Cache Size', 'Hit Rate']].sort_values(by='Cache Size')

        fig, ax = plt.subplots()
        ax.plot(list(map(str, true_df['Cache Size'])), true_df['Hit Rate'], marker="d", label="Dependency Splice")
        ax.plot(list(map(str, false_df['Cache Size'])), false_df['Hit Rate'], marker="o", label="Without Splice")

        p = list(df['Dependent Rule'].drop_duplicates())[0]
        # ax.set_title("Caida trace 6k rules with \n  Dependent Rule: {0}".format(p))

        xy_label_font_size = 28
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)

        ax.set_ylabel('Cache Hit (%)', fontsize=xy_label_font_size)
        ax.set_xlabel("Cache Size", fontsize=xy_label_font_size)
        ax.set_ylim([35, 100])
        ax.legend()
        fig.tight_layout()
        fig.savefig(csv_path.replace('csv', 'jpg'))

        print("s")


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


def main():
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
    MakeRunScript.make_run_sc()


if __name__ == "__main__":
    main()

