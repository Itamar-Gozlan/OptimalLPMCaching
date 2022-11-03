import abc
import ipaddress
import os
import json
from collections import OrderedDict

from Utils import construct_tree
import networkx as nx
import itertools
import copy
import numpy as np
import pandas as pd
import time

ROOT = 0
ROOT_PREFIX = '0.0.0.0/0'
ADD_GOTO_NODES = False


class NodeData:
    def __init__(self, weight=0, size=0, distance=0, n_successors=0, v=0):
        self.subtree_weight = weight
        self.subtree_size = size
        self.subtree_depth = distance
        self.n_successors = n_successors
        self.dependent_set = set()
        self.v = v

    def unpack(self):
        return self.subtree_size, self.n_successors, self.v

    @staticmethod
    def construct_node_data_dict(policy):
        policy = list(map(lambda s: s.strip(), policy))
        policy_tree, rule_to_vertex, successors = HeuristicLPMCache.process_policy(policy)
        vertex_to_rule = {value: key for key, value in rule_to_vertex.items()}
        depth_dict = HeuristicLPMCache.construct_depth_dict(policy_tree)
        # Looking for nodes with low count of successors and big subtrees
        data_dict = {}
        for depth in sorted(list(depth_dict.keys()), reverse=True):
            for v in depth_dict[depth]:
                node_data = NodeData()
                node_data.subtree_size = 1
                node_data.dependent_set = {v}
                if len(successors[v]) == 0:  # leaf
                    node_data.subtree_depth = 1
                else:
                    for u in successors[v]:
                        node_data.subtree_size += data_dict[u].subtree_size
                        node_data.dependent_set.update(data_dict[u].dependent_set)
                    node_data.subtree_depth = 1 + max(
                        [data_dict[u].subtree_depth for u in successors[v]])
                node_data.v = v
                node_data.n_successors = len(successors[v])
                node_data.v = v
                data_dict[v] = node_data

        return data_dict, vertex_to_rule


class FeasibleSet:
    def __init__(self) -> None:
        self.feasible_iset = {}
        self.feasible_iset_weight = {}
        self.item_count = {}

    def insert_iset_item(self, i, item, weight):
        self.feasible_iset[i].add(item)
        self.feasible_iset_weight[i] += weight
        self.item_count[i] += 1

    def superset_of(self, i, subset):
        if not isinstance(self.feasible_iset[i], set):
            self.feasible_iset[i] = set(self.feasible_iset[i])  # dereference
        return set(subset).issubset(self.feasible_iset[i])

    def dereference(self):
        for i in range(len(self.feasible_iset)):
            if not isinstance(self.feasible_iset[i], set):
                self.feasible_iset[i] = set(self.feasible_iset[i])  # dereference

    @staticmethod
    def merge_two_feasible_iset(iset1, iset2, cache_size: int):
        target_feasible_set = FeasibleSet()
        for i in range(cache_size + 1):
            max_weight = 0
            max_j = 0
            for j in range(i + 1):
                if iset1.feasible_iset_weight[j] + iset2.feasible_iset_weight[i - j] > max_weight:
                    max_weight = iset1.feasible_iset_weight[j] + iset2.feasible_iset_weight[i - j]
                    max_j = j
            target_feasible_set.feasible_iset[i] = itertools.chain(iset1.feasible_iset[max_j],
                                                                   iset2.feasible_iset[i - max_j])
            target_feasible_set.item_count[i] = iset1.item_count[max_j] + iset2.item_count[i - max_j]
            target_feasible_set.feasible_iset_weight[i] = max_weight
        target_feasible_set.dereference()
        return target_feasible_set

    @staticmethod
    def OptDTUnion(feasible_set: list, cache_size: int):
        if len(feasible_set) == 1:
            return feasible_set[0]

        split_i = int(len(feasible_set) / 2)
        feasible_iset_x = FeasibleSet.OptDTUnion(feasible_set[:split_i], cache_size)
        feasible_iset_y = FeasibleSet.OptDTUnion(feasible_set[split_i:], cache_size)

        return FeasibleSet.merge_two_feasible_iset(feasible_iset_x, feasible_iset_y, cache_size)


class HeuristicLPMCache:
    def __init__(self, cache_size: int, policy: list, dependency_splice=True):
        self.policy_tree, self.rule_to_vertex, self.successors = HeuristicLPMCache.process_policy(policy)
        self.depth_dict = HeuristicLPMCache.construct_depth_dict(self.policy_tree)
        self.vertex_to_rule = {value: key for key, value in self.rule_to_vertex.items()}
        self.cache_size = cache_size
        self.dependency_splice = dependency_splice
        self.feasible_set = {}
        self.n_goto_nodes = 0
        self.hit_rate = -1

    def compute_cache(self, cache_size, prefix_weight):
        self.cache_size = cache_size
        cache = self.get_cache(prefix_weight)
        self.hit_rate = sum(prefix_weight[v] for v in cache)*100 / sum(prefix_weight.values())

    def get_hit_rate(self):
        return self.hit_rate

    def construct_clean_bottom_up_list(self, cache_candidate_depth_dict, cache):
        all_nodes = list(itertools.chain(
            *[cache_candidate_depth_dict[d] for d in sorted(list(cache_candidate_depth_dict.keys()), reverse=True)]))
        get_vtx = lambda rule: self.rule_to_vertex[rule.split('_')[0]] if 'goto' in rule else self.rule_to_vertex[rule]
        cache_mask = map(get_vtx, cache)
        return list(filter(lambda vtx: vtx not in cache_mask, all_nodes))

    def get_cache(self, prefix_weight):
        cache_candidate_depth_dict = self.get_cache_candidate_tree(prefix_weight)
        self.n_goto_nodes = 0
        for depth in sorted(list(cache_candidate_depth_dict.keys()), reverse=True):
            for v in cache_candidate_depth_dict[depth]:
                self.consider_v_for_cache(v, prefix_weight, self.cache_size)

        return self.return_cache(self.cache_size)

    def return_cache(self, cache_size):
        if ADD_GOTO_NODES:
            get_rule = lambda vtx: self.vertex_to_rule[vtx] if isinstance(vtx, int) else self.vertex_to_rule[int(
                vtx.split("_")[0])] + "_goto"
            self.n_goto_nodes = len(list(filter(lambda rule: isinstance(rule, str) and 'goto' in rule,
                                                self.feasible_set[ROOT].feasible_iset[cache_size])))
        else:
            get_rule = lambda vtx: self.vertex_to_rule[vtx]

        return set([get_rule(v) for v in self.feasible_set[ROOT].feasible_iset[cache_size]])

    def consider_v_for_cache(self, v, prefix_weight, cache_size):
        if len(self.successors[v]) == 0:  # leaf
            return
        else:
            children_feasible_set = FeasibleSet.OptDTUnion(
                self.get_candidate_feasible_set(v, prefix_weight, cache_size),
                cache_size)
            v_in_cache = {i: False for i in range(cache_size + 1)}
            # Adding v
            for i in range(cache_size + 1):
                if children_feasible_set.item_count[i] < i and children_feasible_set.superset_of(i,
                                                                                                 self.successors[
                                                                                                     v]):
                    children_feasible_set.insert_iset_item(i, v, prefix_weight.get(self.vertex_to_rule[v], 0))
                    v_in_cache[i] = True

            if self.dependency_splice and len(self.successors[v]) < cache_size:
                self.apply_dependency_splice(v, prefix_weight, children_feasible_set, v_in_cache, cache_size)
            else:
                self.feasible_set[v] = children_feasible_set


    def get_candidate_feasible_set(self, v, prefix_weight, cache_size):
        # optimization for leafs, power of k choices
        leaf_children_of_v = []
        internal_nodes_children_of_v = []
        for u in self.successors[v]:
            if len(self.successors[u]) == 0:
                leaf_children_of_v.append(u)
            else:
                internal_nodes_children_of_v.append(u)
        sorted_leaf_children_of_v = sorted(leaf_children_of_v,
                                           key=lambda u: prefix_weight.get(self.vertex_to_rule[u], 0), reverse=True)
        leaf_child_feasible_set = FeasibleSet()
        for i in range(cache_size + 1):
            leaf_child_feasible_set.feasible_iset[i] = set(sorted_leaf_children_of_v[:i])
            leaf_child_feasible_set.feasible_iset_weight[i] = sum([prefix_weight.get(self.vertex_to_rule[u], 0) for u in
                                                                   sorted_leaf_children_of_v[:i]])
            leaf_child_feasible_set.item_count[i] = len(sorted_leaf_children_of_v[:i])
        candidate_feasible_set = []
        for u in internal_nodes_children_of_v:
            if self.feasible_set.get(u):
                candidate_feasible_set.append(self.feasible_set.get(u))
        candidate_feasible_set.append(leaf_child_feasible_set)

        return candidate_feasible_set

    def apply_dependency_splice(self, v, prefix_weight, children_feasible_set, v_in_cache, cache_size):
        v_weight = prefix_weight.get(self.vertex_to_rule[v], 0)
        dependency_spliced_v = FeasibleSet()
        spliced = False
        log = []
        for i in range(cache_size + 1):
            j_maybe = i + len(
                set(self.policy_tree.successors(v)) - children_feasible_set.feasible_iset[i])
            if j_maybe + 1 <= cache_size and not v_in_cache[j_maybe + 1] and not v_in_cache[i]:
                sum_j_maybe = children_feasible_set.feasible_iset_weight[j_maybe + 1]
                sum_i_and_v = children_feasible_set.feasible_iset_weight[i] + v_weight
                if sum_j_maybe < sum_i_and_v:
                    spliced = True
                    dependency_spliced_v.feasible_iset_weight[j_maybe + 1] = \
                        children_feasible_set.feasible_iset_weight[i] + v_weight

                    dependency_spliced_v.feasible_iset[j_maybe + 1] = \
                        children_feasible_set.feasible_iset[i].union(set([v]))

                    # Adding 'goto' nodes
                    # Without adding the "goto" nodes
                    if ADD_GOTO_NODES:
                        D = set(map(lambda s: str(s) + "_goto", set(self.policy_tree.successors(v)) -
                                    children_feasible_set.feasible_iset[i]))
                        dependency_spliced_v.feasible_iset[j_maybe + 1] = \
                            dependency_spliced_v.feasible_iset[j_maybe + 1].union(D)

                    dependency_spliced_v.item_count[j_maybe + 1] = j_maybe + 1
        if spliced:
            for i in range(cache_size + 1):
                if dependency_spliced_v.item_count.get(i) is None:  # copy
                    dependency_spliced_v.item_count[i] = children_feasible_set.item_count[i]
                    dependency_spliced_v.feasible_iset[i] = children_feasible_set.feasible_iset[i]
                    dependency_spliced_v.feasible_iset_weight[i] = \
                        children_feasible_set.feasible_iset_weight[i]

            self.feasible_set[v] = dependency_spliced_v
        else:
            self.feasible_set[v] = children_feasible_set

    def OptDTUnion(self, feasible_set, children, prefix_weight, cache_size):
        if len(children) == 1:
            return feasible_set[children[0]]
        split_i = int(len(children) / 2)
        feasible_set_x = self.OptDTUnion(feasible_set, children[:split_i], prefix_weight, cache_size)
        feasible_set_y = self.OptDTUnion(feasible_set, children[split_i:], prefix_weight, cache_size)
        feasible_set = {}
        for i in range(cache_size + 1):
            max_i = set()
            for j in range(i + 1):
                max_i = max(max_i, feasible_set_x[j].union(feasible_set_y[i - j]),
                            key=lambda S: HeuristicLPMCache.set_weight(S, prefix_weight))
            feasible_set[i] = max_i
        return feasible_set

    def get_cache_candidate_tree(self, prefix_weight):
        subtree_weight = {}
        for depth in sorted(list(self.depth_dict.keys()), reverse=True)[:-1]:
            for v in self.depth_dict[depth]:
                subtree_weight[v] = prefix_weight.get(self.vertex_to_rule[v], 0) + \
                                    sum([subtree_weight[u] for u in self.successors[v]])

        non_relevant_root_children = []
        for u in self.policy_tree.neighbors(ROOT):
            if subtree_weight[u] == 0:
                non_relevant_root_children.append(u)

        for u in non_relevant_root_children:
            self.policy_tree.remove_edge(ROOT, u)

        depth_dict = HeuristicLPMCache.construct_depth_dict(self.policy_tree)

        for u in non_relevant_root_children:
            self.policy_tree.add_edge(ROOT, u)

        return depth_dict

    @staticmethod
    def construct_depth_dict(T):
        shortest_path = nx.single_source_shortest_path(T, ROOT)
        depth_dict = {d: [] for d in range(max(map(len, shortest_path.values())) + 1)}
        for v in T.nodes:
            if shortest_path.get(v):
                d = len(shortest_path[v])
                depth_dict[d].append(v)
        return depth_dict

    @staticmethod
    def process_policy(policy):
        longest_prefix_first = sorted(policy, key=lambda v: int(v.split('/')[-1]), reverse=True)
        if ROOT_PREFIX != longest_prefix_first[-1]:
            longest_prefix_first.append(ROOT_PREFIX)
        binary_policy = []
        for prefix in longest_prefix_first:
            ip, mask = prefix.split('/')
            binary_policy.append("{:032b}".format(int(ipaddress.IPv4Address(ip)))[:int(mask)])

        return construct_tree(binary_policy)  # return: T, rule_to_vertex, successors

    @staticmethod
    def set_weight(S, prefix_weight):
        return sum([prefix_weight.get(elem, 0) for elem in S])


# ----------------------- OptimalLPMCache -----------------------
class OptimalLPMCache:
    def __init__(self, policy, prefix_weight, cache_size):
        self.policy_tree, self.rule_to_vertex, self.successors = HeuristicLPMCache.process_policy(policy)
        # optimizing for leafs
        self.successors = {k: sorted(v, key=lambda vx: len(self.successors[vx])) for k, v in self.successors.items()}
        self.vertex_to_rule = {value: key for key, value in self.rule_to_vertex.items()}
        self.depth_dict = HeuristicLPMCache.construct_depth_dict(self.policy_tree)
        self.deg_out = {v: self.policy_tree.out_degree(v) for v in self.policy_tree.nodes}
        self.node_weight = {v: prefix_weight[self.vertex_to_rule[v]] for v in self.policy_tree.nodes}
        self.cache_size = cache_size
        self.vtx_S = {}
        self.S = {}
        self.gtc_nodes = None

    def get_optimal_cache(self):
        for depth in sorted(list(self.depth_dict.keys()), reverse=True)[:-1]:
            for idx, vtx in enumerate(self.depth_dict[depth]):
                self.apply_on_vtx(vtx)

        self.gtc_nodes = self.get_gtc()

    def apply_on_vtx(self, vtx):
        Y_data = {}
        Y_tilde_data = {}
        Y_solution = {}
        Y_tilde_solution = {}
        if self.deg_out[vtx] == 0:
            for i in range(1, self.cache_size + 1):
                Y_data[(vtx, 0, i)] = self.node_weight[vtx]  # S0, S1 for leaf
                Y_tilde_data[(0, 1, i)] = 0  # S0, S1 for leaf
                Y_solution[(vtx, 0, i)] = set([vtx])
                Y_tilde_solution[(0, 1, i)] = set()

        Y_tilde_data, Y_tilde_solution = self.OptDTUnion(self.successors[vtx], vtx)  # (j, r, i)
        for i in range(self.cache_size + 1):
            for r in range(self.deg_out[vtx] + 1):
                j_maybe = i + r + 1  # potential cache size to splice -> Y(x,r,j) is defined empty if not feasible
                if self.deg_out[vtx] + 1 <= j_maybe <= self.cache_size:
                    if Y_tilde_data.get((self.deg_out[vtx], r, i)) is not None:
                        weight_maybe = Y_tilde_data.get((self.deg_out[vtx], r, i), 0) + self.node_weight[vtx]
                        if Y_data.get((vtx, r, j_maybe), 0) < weight_maybe:
                            Y_data[(vtx, r, j_maybe)] = weight_maybe
                            Y_solution[(vtx, r, j_maybe)] = Y_tilde_solution.get((self.deg_out[vtx], r, i),
                                                                                 set()).union(set([vtx]))
        # initialize S(x,0,j), S(x,1,j)
        # initializing with 0, weight of empty set
        S0_weight = {0: None}
        S1_weight = {}
        self.S[vtx] = {0: {}, 1: {}}
        # (vtx, r, i) - (jth child, excluded nodes, cache size)
        for j in range(0, self.cache_size + 1):
            for r in range(0, self.deg_out[vtx] + 1):
                for r_tag in range(self.deg_out[vtx] + 1):
                    if Y_data.get((vtx, r_tag, j), None) is not None and Y_data[(vtx, r_tag, j)] > S0_weight.get(j, 0):
                        S0_weight[j] = Y_data[(vtx, r_tag, j)]
                        self.S[vtx][0][j] = Y_solution.get((vtx, r_tag, j), set())

                    if Y_tilde_data.get((self.deg_out[vtx], r_tag, j), 0) > S1_weight.get(j, -1):
                        S1_weight[j] = Y_tilde_data.get((self.deg_out[vtx], r_tag, j), 0)
                        self.S[vtx][1][j] = Y_tilde_solution.get((self.deg_out[vtx], r_tag, j), set())

        for i in range(self.cache_size):
            if i not in S0_weight:
                S0_weight[i] = None
            if i not in S1_weight:
                S1_weight[i] = None

        self.vtx_S[vtx] = [S0_weight, S1_weight]

    # -------------------------- OptDTUnion --------------------------

    def OptDTUnion(self, children_array, vtx):  # return (m,k) collection of SplicingFeasble sets
        # (j,r,i) - Optimal weight of T(<=j) with excluding r vertices
        Y_solution = {}
        Y_data = {}
        for j in range(1, self.deg_out[vtx] + 1):  # j=1,..,m -> extend solution to include Ty
            for r in range(0, j + 1):  # number of roots to exclude
                for i in range(0, self.cache_size + 1):  # possible cache size
                    max_weight, (i_star, r_star, j_star) = self.OptDTUnion_it(Y_data, j, r, i, children_array)
                    if max_weight >= 0:
                        Y_data[(j, r, i)] = max_weight
                        self.OptDTUnion_update_max_weight_solution(j, r, i, (i_star, r_star, j_star), Y_solution,
                                                                   children_array, max_weight, Y_data)

        return Y_data, Y_solution  # by reference

    def OptDTUnion_it(self, Y_data, j, r, i, children_array):
        max_weight = -1
        weight_with_S0 = None
        weight_with_S1 = None
        (i_star, r_star, j_star) = (-1, -1, -1)

        latex_S = lambda j, r, i, i_t, s: "Y^{" + str(j - 1) + "," + str(r - 1) + "}_{" + str(
            i_t) + "} \\cup " + "S^{" + str(j) + "," + str(s) + "}_{" + str(i - i_t) + "}"
        case_log = []
        for i_t in range(i + 1):
            S0j_imi_t = self.vtx_S[children_array[j - 1]][0].get(i - i_t, 0)  # S0
            S1j_imi_t = self.vtx_S[children_array[j - 1]][1].get(i - i_t, 0)  # S1
            if j == 1:  # initialization
                if r - 1 >= 0:
                    weight_with_S1 = S1j_imi_t
                    case_log.append((latex_S(j, r, i, i_t, 1), weight_with_S1))
                if r < j and S0j_imi_t is not None:
                    weight_with_S0 = S0j_imi_t
                    case_log.append((latex_S(j, r, i, i_t, 0), weight_with_S0))
            else:  # step
                Y_data_jm1_rm1 = Y_data.get((j - 1, r - 1, i_t))  # avoid extra get
                if r - 1 >= 0 and Y_data_jm1_rm1 is not None:
                    weight_with_S1 = Y_data_jm1_rm1 + S1j_imi_t
                    case_log.append((latex_S(j, r, i, i_t, 1), weight_with_S1))

                Y_data_jm1_r = Y_data.get((j - 1, r, i_t))
                if r < j and Y_data_jm1_r is not None and S0j_imi_t is not None:
                    weight_with_S0 = Y_data_jm1_r + S0j_imi_t
                    case_log.append((latex_S(j, r, i, i_t, 0), weight_with_S0))

            if weight_with_S1 is None and weight_with_S0 is None:
                continue
            if weight_with_S0 is not None and weight_with_S1 is not None:
                if weight_with_S1 > max_weight and weight_with_S1 > weight_with_S0:
                    max_weight = weight_with_S1
                    (i_star, r_star, j_star) = (i_t, 0, j)
                if weight_with_S0 > max_weight and weight_with_S0 >= weight_with_S1:
                    max_weight = weight_with_S0
                    (i_star, r_star, j_star) = (i_t, 1, j)
            elif weight_with_S0 is not None:
                if weight_with_S0 > max_weight:
                    max_weight = weight_with_S0
                    (i_star, r_star, j_star) = (i_t, 1, j)
            elif weight_with_S1 is not None:
                if weight_with_S1 > max_weight:
                    max_weight = weight_with_S1
                    (i_star, r_star, j_star) = (i_t, 0, j)

        return max_weight, (i_star, r_star, j_star)

    def OptDTUnion_update_max_weight_solution(self, j, r, i, star, Y_solution, children_array, max_weight, Y_data):
        get_Sj0i = lambda j_child, idx: self.S.get(children_array[j_child - 1], {}).get(0, {}).get(idx,
                                                                                                   set())
        get_Sj1i = lambda j_child, idx: self.S.get(children_array[j_child - 1], {}).get(1, {}).get(idx,
                                                                                                   set())
        (i_star, r_star, j_star) = star
        if r_star == 1:  # sj0(i-i_star)
            Y_solution[(j, r, i)] = Y_solution.get((j - 1, r, i_star), set()).union(
                get_Sj0i(j, i - i_star))
            weight_with_S0_t = Y_data.get((j - 1, r, i_star), 0) + \
                               self.vtx_S[children_array[j - 1]][0].get(i - i_star, 0)
            if weight_with_S0_t != max_weight:
                print('Err')
        else:
            Y_solution[(j, r, i)] = Y_solution.get((j - 1, r - 1, i_star), set()).union(get_Sj1i(j, i - i_star))
            weight_with_S1_t = Y_data.get((j - 1, r - 1, i_star), 0) + \
                               self.vtx_S[children_array[j - 1]][1].get(i - i_star, 0)
            if weight_with_S1_t != max_weight:
                print('Err')

    def get_gtc(self):
        gtc_nodes = set()
        for vtx in self.S[ROOT][1][self.cache_size]:
            for gtc in set(self.policy_tree.neighbors(vtx)) - self.S[ROOT][1][self.cache_size]:
                gtc_nodes.add(gtc)
        return gtc_nodes

    # -------------------------- Analyze Solution --------------------------
    @staticmethod
    def Y_data_j_to_df(Y_data, j, cache_size):
        np_data_2d = np.zeros((cache_size + 1, j + 1))
        for r in range(j + 1):
            for i in range(cache_size + 1):
                np_data_2d[i, r] = Y_data.get((j, r, i), -1)

        return np_data_2d

    @staticmethod
    def solution_to_df(solution, j, cache_size):
        return pd.DataFrame([[str(solution.get((j, r, i), '{}')) for r in range(j + 1)] for i in range(cache_size + 1)])

    @staticmethod
    def Y_data_2D_to_DF(Y_data, j, cache_size):
        np_data_2d = np.zeros((cache_size + 1, j + 1))
        for r in range(j + 1):
            for i in range(cache_size + 1):
                np_data_2d[i, r] = Y_data.get((j, i), -1)

        return np_data_2d


class OptimizedOptimalLPMCache:
    """
    Ideas:
    Memory optimization - delete pred depth data structure
    cut loops
    Don't run OptDTUnion of leaves
    Don't keep 0 results
    """

    def __init__(self, policy, prefix_weight, cache_size, logfile_path=None):
        self.policy_tree, self.rule_to_vertex, self.successors = HeuristicLPMCache.process_policy(policy)
        # optimizing for leafs
        self.successors = {k: sorted(v, key=lambda vx: len(self.successors[vx])) for k, v in self.successors.items()}
        self.vertex_to_rule = {value: key for key, value in self.rule_to_vertex.items()}
        self.depth_dict = HeuristicLPMCache.construct_depth_dict(self.policy_tree)
        self.deg_out = {v: self.policy_tree.out_degree(v) for v in self.policy_tree.nodes}
        self.node_weight = {v: prefix_weight[self.vertex_to_rule[v]] for v in self.policy_tree.nodes}
        self.cache_size = cache_size
        self.vtx_S = {}
        self.S = {}
        self.hit_rate = -1
        if logfile_path:
            self.logfile = open(logfile_path, 'w+')
        else:
            self.logfile = None

    def compute_cache(self, cache_size, prefix_weight):
        self.cache_size = cache_size
        self.get_optimal_cache(cache_size)
        total_weight = sum(self.node_weight.values())
        self.hit_rate = 100 * self.vtx_S[ROOT][1][cache_size] / total_weight

    def get_hit_rate(self):
        return self.hit_rate

    def get_optimal_cache(self, cache_size = None):
        # import cProfile, pstats
        # profiler = cProfile.Profile()
        # profiler.enable()
        t0 = time.time()
        for depth in sorted(list(self.depth_dict.keys()), reverse=True)[:-1]:
            for idx, vtx in enumerate(self.depth_dict[depth]):
                self.apply_on_vtx(vtx)

    def apply_on_vtx(self, vtx):
        Y_data = {}
        Y_tilde_data = {}
        if self.deg_out[vtx] == 0:
            for i in range(1, self.cache_size + 1):
                Y_data[(vtx, 0, i)] = self.node_weight[vtx]  # S0, S1 for leaf
                Y_tilde_data[(0, 1, i)] = 0  # S0, S1 for leaf

        else:  # vtx != leaf
            if vtx != ROOT:
                Y_tilde_data = self.OptDTUnion(self.successors[vtx], vtx)  # (j, r, i)
                for i in range(self.cache_size + 1):
                    for r in range(self.deg_out[vtx] + 1):
                        j_maybe = i + r + 1  # potential cache size to splice -> Y(x,r,j) is defined empty if not feasible
                        if self.deg_out[vtx] + 1 <= j_maybe <= self.cache_size:
                            if Y_tilde_data.get((self.deg_out[vtx], r, i)) is not None:
                                weight_maybe = Y_tilde_data.get((self.deg_out[vtx], r, i), 0) + self.node_weight[vtx]
                                if Y_data.get((vtx, r, j_maybe), 0) < weight_maybe:
                                    Y_data[(vtx, r, j_maybe)] = weight_maybe

            else:  # vtx == ROOT
                Y_data = self.GreedyDTUnion(self.successors[vtx], vtx)  # (j, i)
                S1_weight = {}
                S0_weight = {}
                for i in range(self.cache_size + 1):
                    S1_weight[i] = Y_data.get((self.deg_out[ROOT], i), 0)
                self.vtx_S[vtx] = [S0_weight, S1_weight]
                return

        # initialize S(x,0,j), S(x,1,j)
        # initializing with 0, weight of empty set
        S0_weight = {0: None}
        S1_weight = {}
        self.S[vtx] = {0: {}, 1: {}}
        # (vtx, r, i) - (jth child, excluded nodes, cache size)
        for j in range(0, self.cache_size + 1):
            for r in range(0, self.deg_out[vtx] + 1):
                for r_tag in range(self.deg_out[vtx] + 1):
                    if Y_data.get((vtx, r_tag, j), None) is not None and Y_data[(vtx, r_tag, j)] > S0_weight.get(j, 0):
                        S0_weight[j] = Y_data[(vtx, r_tag, j)]

                    if Y_tilde_data.get((self.deg_out[vtx], r_tag, j), 0) > S1_weight.get(j, -1):
                        S1_weight[j] = Y_tilde_data.get((self.deg_out[vtx], r_tag, j), 0)

        for i in range(self.cache_size):
            if i not in S0_weight:
                S0_weight[i] = None
            if i not in S1_weight:
                S1_weight[i] = None

        self.vtx_S[vtx] = [S0_weight, S1_weight]

    # -------------------------- OptDTUnion --------------------------

    def OptDTUnion(self, children_array, vtx):  # return (m,k) collection of SplicingFeasble sets
        # (j,r,i) - Optimal weight of T(<=j) with excluding r vertices
        Y_data, n_child_leafs = self.calculate_leafs_Y_data(children_array)

        # print("Non leafs children: {0}".format(self.deg_out[vtx] - n_child_leafs))
        for j in range(n_child_leafs + 1, self.deg_out[vtx] + 1):  # j=1,..,m -> extend solution to include Ty
            for r in range(0, j + 1):  # number of roots to exclude
                cache_size_gain = True
                for i in range(0, self.cache_size + 1):  # possible cache size
                    if cache_size_gain:
                        max_weight, (i_star, r_star, j_star) = self.OptDTUnion_it(Y_data, j, r, i, children_array)
                    if max_weight >= 0:
                        Y_data[(j, r, i)] = max_weight
                        if i > 1 and Y_data.get((j, r, i - 1), -1) == Y_data[(j, r, i)]:
                            cache_size_gain = False
                        # No extra gain from increasing cache size, we can stop calling self.OptDTUnion_it

            if j - 2 > 0:  # memory optimization: remove j-1
                for r in range(0, j - 2):  # number of roots to exclude
                    for i in range(0, self.cache_size + 1):  # possible cache size
                        if (j - 2, r, i) in Y_data:
                            del Y_data[(j - 2, r, i)]

        return Y_data  # by reference

    def calculate_leafs_Y_data(self, children_array):
        Y_data = {}  # (j,r,i)
        n_child_leafs = 0
        while len(children_array) > n_child_leafs and len(self.successors[children_array[n_child_leafs]]) == 0:
            n_child_leafs += 1

        sorted_leafs_weight = sorted([self.node_weight[v] for v in children_array[:n_child_leafs]], reverse=True)
        for r in range(0, n_child_leafs + 1):  # number of roots to exclude
            for i in range(0, self.cache_size + 1):  # possible cache size
                leafs_to_choose = n_child_leafs - r
                if leafs_to_choose <= i:
                    Y_data[(n_child_leafs, r, i)] = sum(sorted_leafs_weight[:leafs_to_choose])
        return Y_data, n_child_leafs

    def OptDTUnion_it(self, Y_data, j, r, i, children_array):
        max_weight = -1
        weight_with_S0 = None
        weight_with_S1 = None
        (i_star, r_star, j_star) = (-1, -1, -1)
        for i_t in range(i + 1):
            S0j_imi_t = self.vtx_S[children_array[j - 1]][0].get(i - i_t, 0)  # S0
            S1j_imi_t = self.vtx_S[children_array[j - 1]][1].get(i - i_t, 0)  # S1
            if j == 1:  # initialization
                if r - 1 >= 0:
                    weight_with_S1 = S1j_imi_t
                if r < j and S0j_imi_t is not None:
                    weight_with_S0 = S0j_imi_t
            else:  # step
                Y_data_jm1_rm1 = Y_data.get((j - 1, r - 1, i_t))  # avoid extra get
                if r - 1 >= 0 and Y_data_jm1_rm1 is not None:
                    weight_with_S1 = Y_data_jm1_rm1 + S1j_imi_t

                Y_data_jm1_r = Y_data.get((j - 1, r, i_t))
                if r < j and Y_data_jm1_r is not None and S0j_imi_t is not None:
                    weight_with_S0 = Y_data_jm1_r + S0j_imi_t

            if weight_with_S1 is None and weight_with_S0 is None:
                continue
            if weight_with_S0 is not None and weight_with_S1 is not None:
                if weight_with_S1 > max_weight and weight_with_S1 > weight_with_S0:
                    max_weight = weight_with_S1
                    (i_star, r_star, j_star) = (i_t, 0, j)
                if weight_with_S0 > max_weight and weight_with_S0 >= weight_with_S1:
                    max_weight = weight_with_S0
                    (i_star, r_star, j_star) = (i_t, 1, j)
            elif weight_with_S0 is not None:
                if weight_with_S0 > max_weight:
                    max_weight = weight_with_S0
                    (i_star, r_star, j_star) = (i_t, 1, j)
            elif weight_with_S1 is not None:
                if weight_with_S1 > max_weight:
                    max_weight = weight_with_S1
                    (i_star, r_star, j_star) = (i_t, 0, j)

        return max_weight, (i_star, r_star, j_star)

    # -------------------------- GreedyDTUnion --------------------------

    def GreedyDTUnion(self, children_array, vtx):
        Y_data = {}  # (j,r,i)
        n_child_leafs = 0
        while len(children_array) > n_child_leafs and len(self.successors[children_array[n_child_leafs]]) == 0:
            n_child_leafs += 1

        # print("GreedyDTUnion: Non leafs children: {0}".format(self.deg_out[vtx] - n_child_leafs))

        sorted_leafs_weight = sorted([self.node_weight[v] for v in children_array[:n_child_leafs]], reverse=True)
        for i in range(0, self.cache_size + 1):  # possible cache size
            Y_data[(n_child_leafs, i)] = sum(sorted_leafs_weight[:i])

        for j in range(n_child_leafs + 1, self.deg_out[vtx] + 1):
            cache_size_gain = True
            for i in range(0, self.cache_size + 1):
                max_weight = -1
                if cache_size_gain:
                    max_weight = self.GreedyDTUnion_it(Y_data, j, i, children_array)

                if max_weight > 0:
                    Y_data[(j, i)] = max_weight

            if j - 2 > 0:  # memory optimization: remove j-1
                for r in range(0, j - 2):  # number of roots to exclude
                    for i in range(0, self.cache_size + 1):  # possible cache size
                        if (j - 2, r, i) in Y_data:
                            del Y_data[(j - 2, r, i)]

        return Y_data

    def GreedyDTUnion_it(self, Y_data, j, i, children_array):
        max_weight = -1
        for i_t in range(i + 1):
            max_S_j = -1
            S0j_imi_t = self.vtx_S[children_array[j - 1]][0].get(i - i_t, 0)  # S0
            S1j_imi_t = self.vtx_S[children_array[j - 1]][1].get(i - i_t, 0)  # S1
            if S0j_imi_t is None and S1j_imi_t is None:
                continue
            if S0j_imi_t is not None and S1j_imi_t is not None:
                max_S_j = max(S0j_imi_t, S1j_imi_t)

            elif S0j_imi_t is not None and S1j_imi_t is None:
                max_S_j = S0j_imi_t

            elif S0j_imi_t is None and S1j_imi_t is not None:
                max_S_j = S1j_imi_t
            if max_S_j + Y_data.get((j - 1, i_t), 0) > max_weight:
                max_weight = max_S_j + Y_data.get((j - 1, i_t), 0)

        return max_weight

    def to_json(self, dir_path, elapsed_time):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(dir_path + '/vtx_S.json', 'w') as f:
            json.dump(self.vtx_S[0][1], f)
        with open(dir_path + '/vertex_to_rule.json', 'w') as f:
            json.dump(self.vertex_to_rule, f)
        total_weight = sum(self.node_weight.values())
        df = pd.DataFrame(columns=["Cache Size", "Hit Rate", "Elapsed Time"])
        for i in range(self.cache_size + 1):
            row = {"Cache Size": i,
                   "Hit Rate": 100 * self.vtx_S[ROOT][1][i] / total_weight,
                   "Elapsed Time": elapsed_time}
            df = df.append(row, ignore_index=True)
        df.to_csv(dir_path + '/result.csv')



class CacheFlow:
    class NodeM:
        def __init__(self):
            self.set = None
            self.cost = None
            self.weight = None
            self.ratio = None

        def __lt__(self, other):
            return self.ratio < other.ratio

        def __le__(self, other):
            return self.ratio <= other.ratio

        def __eq__(self, other):
            return self.ratio == other.ratio

        def __gt__(self, other):
            return self.ratio > other.ratio

        def __ge__(self, other):
            return self.ratio >= other.ratio

    def __init__(self, policy, prefix_weight):
        self.policy_tree, self.rule_to_vertex, self.successors = HeuristicLPMCache.process_policy(policy)
        data_dict, self.vertex_to_rule = NodeData.construct_node_data_dict(policy)
        self.M = {}
        for v in self.policy_tree.nodes:
            data = data_dict[v]
            self.M[(v, 'dependent')] = CacheFlow.NodeM()
            self.M[(v, 'dependent')].set = data.dependent_set
            self.M[(v, 'dependent')].cost = len(data.dependent_set)
            self.M[(v, 'dependent')].weight = sum([prefix_weight[self.vertex_to_rule[u]] for u in data.dependent_set])
            self.M[(v, 'dependent')].ratio = self.M[(v, 'dependent')].weight/self.M[(v, 'dependent')].cost

            self.M[(v, 'cover')] = CacheFlow.NodeM()
            self.M[(v, 'cover')].set = set(self.policy_tree.neighbors(v)).union([v])
            self.M[(v, 'cover')].cost = len(self.M[(v, 'cover')].set)
            self.M[(v, 'cover')].weight = prefix_weight[self.vertex_to_rule[v]]
            self.M[(v, 'cover')].ratio = self.M[(v, 'cover')].weight/self.M[(v, 'cover')].cost

        self.hit_rate = -1

    def GTC(self, v):
        return str(v) + "_gtc"

    def MixedSet(self, k):
        cache = set()
        while len(cache) < k and len(self.M) > 0:
            for (v, type), node_m in sorted(self.M.items(), key=lambda x: x[1], reverse=True):
                if type == 'dependent' and len(cache - set(self.GTC(v))) + node_m.cost <= k:
                    if self.GTC(v) in cache:
                        cache.remove(self.GTC(v))
                    cache.add(v)
                    for type in ['dependent', 'cover']:
                        for u in node_m.set - set([v]):
                            cache.add(u)
                            del self.M[(u, type)]
                    break
                if type == 'cover' and len(cache - set(self.GTC(v))) + node_m.cost <= k:
                    if self.GTC(v) in cache:
                        cache.remove(self.GTC(v))
                    cache.add(v)
                    for u in node_m.set - set([v]):
                        cache.add(self.GTC((u)))
                    break
            x_pred = list(self.policy_tree.predecessors(v))[0]  # tree, one element is predecessor

            if x_pred != ROOT and x_pred not in cache:
                self.M[(x_pred, 'cover')].set = self.M[(x_pred, 'cover')].set - self.M[(v, 'cover')].set
                self.M[(x_pred, 'cover')].cost = self.M[(x_pred, 'cover')].cost - 1
                self.M[(x_pred, 'cover')].ratio = self.M[(x_pred, 'cover')].weight / self.M[(x_pred, 'cover')].cost

            while x_pred != ROOT and x_pred not in cache:
                self.M[(x_pred, 'dependent')].set = self.M[(x_pred, 'dependent')].set - self.M[(v, 'dependent')].set
                self.M[(x_pred, 'dependent')].cost = self.M[(x_pred, 'dependent')].cost - self.M[(v, 'dependent')].cost
                self.M[(x_pred, 'dependent')].weight = self.M[(x_pred, 'dependent')].weight - self.M[(v, 'dependent')].weight
                self.M[(x_pred, 'dependent')].ratio = self.M[(x_pred, 'dependent')].weight / self.M[(x_pred, 'dependent')].cost
                x_pred = list(self.policy_tree.predecessors(x_pred))[0]

            for type in ['dependent', 'cover']:
                del self.M[(v, type)]

        return cache

    def compute_cache(self, cache_size, prefix_weight):
        self.__init__(prefix_weight.keys(), prefix_weight)
        cache = self.MixedSet(cache_size)
        cache = filter(lambda elem: isinstance(elem, int), cache)
        sum_total = sum(prefix_weight.values())
        sum_all_nodes = sum([prefix_weight[self.vertex_to_rule[u]] for u in cache])
        self.hit_rate = (sum_all_nodes) / sum_total

    def get_hit_rate(self):
        return self.hit_rate