import abc
import ipaddress
from Utils import construct_tree
import networkx as nx
import itertools
import copy
import numpy as np
import pandas as pd

ROOT = 0
ROOT_PREFIX = '0.0.0.0/0'
ADD_GOTO_NODES = True


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


class HeuristicLPMCache():
    def __init__(self, cache_size: int, policy: list, dependency_splice=True):
        self.policy_tree, self.rule_to_vertex, self.successors = HeuristicLPMCache.process_policy(policy)
        self.depth_dict = HeuristicLPMCache.construct_depth_dict(self.policy_tree)
        self.vertex_to_rule = {value: key for key, value in self.rule_to_vertex.items()}
        self.cache_size = cache_size
        self.dependency_splice = dependency_splice
        self.feasible_set = {}
        self.n_goto_nodes = 0

    def construct_clean_bottom_up_list(self, cache_candidate_depth_dict, cache):
        all_nodes = list(itertools.chain(
            *[cache_candidate_depth_dict[d] for d in sorted(list(cache_candidate_depth_dict.keys()), reverse=True)]))
        get_vtx = lambda rule: self.rule_to_vertex[rule.split('_')[0]] if 'goto' in rule else self.rule_to_vertex[rule]
        cache_mask = map(get_vtx, cache)
        return list(filter(lambda vtx: vtx not in cache_mask, all_nodes))

    def get_cache(self, prefix_weight):
        cache_candidate_depth_dict = self.get_cache_candidate_tree(prefix_weight)
        print("SUM: {0}".format(sum([len(l) for l in cache_candidate_depth_dict.values()])))
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

        print("goto_nodes: {0}".format(self.n_goto_nodes))
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
        for i in range(cache_size + 1):
            j_maybe = i + len(
                set(self.policy_tree.successors(v)) - children_feasible_set.feasible_iset[i])
            if j_maybe + 1 <= cache_size and not v_in_cache[j_maybe + 1]:
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


class OnlineTreeCache:
    def __init__(self, policy, cache_size):
        self.subtree_size = None
        self.policy_tree, self.rule_to_vertex, self.successors = HeuristicLPMCache.process_policy(policy)
        self.vertex_to_rule = {value: key for key, value in self.rule_to_vertex.items()}
        self.construct_subtree_size()
        # for parent, descendant_array in self.successors.items():
        #     for descendant in descendant_array:
        #         self.predecessor[descendant] = parent
        self.alpha = 2
        self.subtree_weight = {}
        self.cache_size = cache_size
        self.cache = set()
        self.rule_counter = {}

    def cache_miss(self, rule):
        vtx = self.rule_to_vertex[rule]
        if self.subtree_size[vtx] <= self.cache_size:  # no need to keep track on subforests larger than cache size
            self.subtree_weight[vtx] = 1 + self.subtree_weight.get(vtx, 0)
            self.update_subtree_weight(vtx, 1)
            self.update_cache(vtx)
            self.rule_counter[vtx] = 1 + self.rule_counter.get(vtx, 0)

    def cache_hit(self, rule):
        vtx = self.rule_to_vertex[rule]
        self.update_subtree_weight(vtx, -1)
        self.rule_counter[vtx] = -1 + self.rule_counter.get(vtx, 0)

    def construct_subtree_size(self):
        self.subtree_size = {}
        depth_dict = HeuristicLPMCache.construct_depth_dict(self.policy_tree)
        for d in sorted(list(depth_dict.keys()), reverse=True):
            for vtx in depth_dict[d]:
                self.subtree_size[vtx] = 1
                for sucessor in self.successors[vtx]:
                    self.subtree_size[vtx] += self.subtree_size[sucessor]

    def update_subtree_weight(self, vtx, weight):
        if weight == 0:
            return
        while self.subtree_size[vtx] <= self.cache_size and len(list(self.policy_tree.predecessors(vtx))) > 0:
            predecessor = list(self.policy_tree.predecessors(vtx))[0]  # one predecessor in a tree
            self.subtree_weight[predecessor] = weight + self.subtree_weight.get(predecessor, 0)
            vtx = predecessor

    def update_cache(self, vtx):
        # saturation: cnt_t(X) >= |X|*alpha
        if self.subtree_weight[vtx] >= self.subtree_size[vtx] * self.alpha:
            # maximality: cnt_t(Y) <= |Y|*alpha for any Y strict superset of X
            pred_vtx = list(self.policy_tree.predecessors(vtx))[0]  # one predecessor in a tree
            while self.subtree_size[pred_vtx] <= self.cache_size and self.subtree_weight[pred_vtx] >= self.subtree_size[
                pred_vtx] * self.alpha:
                vtx = pred_vtx
                if len(list(self.policy_tree.predecessors(vtx))) > 0:
                    pred_vtx = list(self.policy_tree.predecessors(vtx))[0]  # one predecessor in a tree
                else:
                    break

            positive_change_set = set([self.vertex_to_rule[u] for u in nx.descendants(self.policy_tree, vtx)])
            positive_change_set.add(self.vertex_to_rule[vtx])
            # Return all nodes reachable from \(source\) in G.
            self.cache = self.cache.union(positive_change_set)
            if len(self.cache) > self.cache_size:  # flush
                for rule_to_zero in self.cache - positive_change_set:
                    v = self.rule_to_vertex[rule_to_zero]
                    self.update_subtree_weight(v, -self.rule_counter.get(v, 0))  # increasing the subtree weight
                    self.rule_counter[v] = 0
                self.cache = positive_change_set

            # TODO - also to the nodes in the cache
            self.update_subtree_weight(vtx, -self.subtree_weight[vtx])  # change to zero
            for v in positive_change_set:
                self.subtree_weight[v] = 0
                self.rule_counter[v] = 0


# ----------------------- OptimalLPMCache -----------------------
"""
r - root to exclude
"""


class OptimalLPMCache:
    """
    Ideas:
    Memory optimization - delete pred depth data structure
    cut loops
    Don't run OptDTUnion of leaves
    Don't keep 0 results
    """

    def __init__(self, policy, prefix_weight, cache_size):
        self.policy_tree, self.rule_to_vertex, self.successors = HeuristicLPMCache.process_policy(policy)
        self.vertex_to_rule = {value: key for key, value in self.rule_to_vertex.items()}
        self.depth_dict = HeuristicLPMCache.construct_depth_dict(self.policy_tree)
        self.deg_out = {v: self.policy_tree.out_degree(v) for v in self.policy_tree.nodes}
        self.node_weight = {v: prefix_weight[self.vertex_to_rule[v]] for v in self.policy_tree.nodes}
        self.cache_size = cache_size
        self.vtx_S = {}
        self.vtx_Y = {}
        self.vtx_tilde_Y = {}
        self.S_r = {}
        self.S = {}
        self.gtc_nodes = None

    def OptDTUnion_it(self, Y_data, j, r, i, children_array):
        # case_log = []
        max_weight = -1
        weight_with_S0 = 0
        weight_with_S1 = 0
        (i_star, r_star, j_star) = (-1, -1, -1)
        for i_t in range(i + 1):
            """
            Should be applied only to potential (i,r)-feasible sets
            S(j, 0, i-it) union Y_data(j-1,r, it) is defined if:
            1. j == 1 and S(j, 0, i-it) > 0
            2. S(j, 0, i-it) > 0 and Y_data(j-1,r, it) > 0
            S(j, 1, i-it) union Y_data(j-1,r-1, it) is defined if:
            1. j == 1 and S(j, 1, i-it) > 0
            2. Y_data(j-1,r, it) > 0 because root has to come from j-1
            
            if we don't take all cache from S1 (i_t > 0) we have to make sure Y(j-1,r-1,i_t) is feasible
            """
            S0j_imi_t = self.vtx_S[children_array[j - 1]][0].get(i - i_t, 0) # S0
            S1j_imi_t = self.vtx_S[children_array[j - 1]][1].get(i - i_t, 0) # S1
            if r - 1 >= 0 and ((j == 1) or Y_data.get((j - 1, r - 1, i_t), 0) > 0): # Y(j-1, r-1, i_t) union S1j_imi_t
                weight_with_S1 = Y_data.get((j - 1, r - 1, i_t), 0) + S1j_imi_t

            """
            Y(j,r,i) = Y(j-1, r, i_t) U S0j(i-it) is feasible if:
            1. r < j. If r == j can't take root
            2. j == 1 and Y(j-1, r, i_t) = empty_set 
            """
            if r < j and ((j == 1) or Y_data.get((j - 1, r, i_t), 0) or r == 0):
                weight_with_S0 = Y_data.get((j - 1, r, i_t), 0) + S0j_imi_t
            if weight_with_S1 > max_weight and weight_with_S1 > weight_with_S0:
                max_weight = weight_with_S1
                (i_star, r_star, j_star) = (i_t, 0, j)
            if weight_with_S0 > max_weight and weight_with_S0 >= weight_with_S1:
                max_weight = weight_with_S0
                (i_star, r_star, j_star) = (i_t, 1, j)

            # if r == 0:  # can only consider S0
            #     if (j == 1 and S0j_imi_t > 0) or (S0j_imi_t > 0 and Y_data.get((j - 1, r, i_t), 0) > 0):
            #         weight_with_S0 = Y_data.get((j - 1, r, i_t), 0) + S0j_imi_t
            #     # case_log.append((0, weight_with_S0))
            # elif r == j:  # can only consider S1
            #     if (j == 1 and S1j_imi_t > 0) or (Y_data.get((j - 1, r - 1, i_t), 0) > 0) or (i_t == 0 and S1j_imi_t > 0):
            #         weight_with_S1 = Y_data.get((j - 1, r - 1, i_t), 0) + S1j_imi_t
            #     # case_log.append((1, weight_with_S1))
            #
            # else:
            #     # (i,r) - feasible set with j-1
            #     if (j == 1 and S0j_imi_t > 0) or (S0j_imi_t > 0 and Y_data.get((j - 1, r, i_t), 0) > 0):
            #         weight_with_S0 = Y_data.get((j - 1, r, i_t), 0) + S0j_imi_t
            #
            #     # case_log.append((2, weight_with_S0))
            #     # (i,r) - feasible set without j-1
            #     if (j == 1 and S1j_imi_t > 0) or (Y_data.get((j - 1, r - 1, i_t), 0) > 0):
            #         weight_with_S1 = Y_data.get((j - 1, r - 1, i_t), 0) + S1j_imi_t
            #
            #     # case_log.append((3, weight_with_S1))
            # if weight_with_S0 > max_weight and weight_with_S0 >= weight_with_S1:
            #     max_weight = weight_with_S0
            #     (i_star, r_star, j_star) = (i_t, 1, j)
            #
            # if weight_with_S1 > max_weight and weight_with_S1 > weight_with_S0:
            #     max_weight = weight_with_S1
            #     (i_star, r_star, j_star) = (i_t, 0, j)
        return max_weight, (i_star, r_star, j_star)

    def OptDTUnion_update_max_weight_solution(self, j, r, i, star, Y_solution, children_array, max_weight, Y_data):
        # if 24 in children_array and (j,r,i) == (2, 1, 4):
        #     print("missing 18")
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

    def OptDTUnion(self, children_array, vtx):  # return (m,k) collection of SplicingFeasble sets
        # (j,r,i) - Optimal weight of T(<=j) with excluding r vertices
        Y_data = {}  # (j,r,i)
        Y_solution = {}
        for j in range(1, self.deg_out[vtx] + 1):  # j=1,..,m -> extend solution to include Ty
            for r in range(0, j + 1):  # number of roots to exclude
                for i in range(0, self.cache_size + 1):  # possible cache size
                    if vtx == 24 and (j,r,i) == (2,1,1):
                         print("Y(2,1,1) need to be 0")
                    max_weight, (i_star, r_star, j_star) = self.OptDTUnion_it(Y_data, j, r, i, children_array)
                    if max_weight > 0:
                        Y_data[(j, r, i)] = max_weight
                        self.OptDTUnion_update_max_weight_solution(j, r, i, (i_star, r_star, j_star), Y_solution,
                                                                   children_array, max_weight, Y_data)

            # TODO: memory optimization: remove j-1
        return Y_data, Y_solution  # by reference

    def apply_on_vtx(self, vtx):
        Y_data = {}
        Y_tilde_data = {}
        Y_solution = {}
        if self.deg_out[vtx] == 0:
            for i in range(1, self.cache_size + 1):
                Y_data[(vtx, 0, i)] = self.node_weight[vtx]  # S0, S1 for leaf
                Y_solution[(vtx, 0, i)] = set([vtx])

        else:  # vtx != leaf
            Y_tilde_data, Y_tilde_solution = self.OptDTUnion(self.successors[vtx], vtx)  # (j, r, i)
            for i in range(self.cache_size + 1):
                for r in range(self.deg_out[vtx] + 1):
                    j_maybe = i + r + 1  # potential cache size to splice -> Y(x,r,j) is defined empty if not feasible
                    if self.deg_out[vtx] < j_maybe <= self.cache_size:
                        weight_maybe = Y_tilde_data.get((self.deg_out[vtx], r, i), 0) + self.node_weight[vtx]
                        if Y_data.get((vtx, r, j_maybe), 0) < weight_maybe:
                            Y_data[(vtx, r, j_maybe)] = weight_maybe
                            Y_solution[(vtx, r, j_maybe)] = Y_tilde_solution.get((self.deg_out[vtx], r, i),
                                                                                 set()).union(set([vtx]))

        # initialize S(x,0,j), S(x,1,j)
        # initializing with 0, weight of empty set
        S0_weight = {0: 0}
        S1_weight = {0: 0}
        self.S[vtx] = {0: {}, 1: {}}
        self.S_r[vtx] = [None, None]
        # (vtx, r, i) - (jth child, excluded nodes, cache size)
        for j in range(0, self.cache_size + 1):
            for r in range(0, self.deg_out[vtx] + 1):
                for r_tag in range(self.deg_out[vtx] + 1):
                    if Y_data.get((vtx, r_tag, j), 0) > S0_weight.get(j, 0):
                        S0_weight[j] = Y_data[(vtx, r_tag, j)]
                        self.S[vtx][0][j] = Y_solution.get((vtx, r_tag, j), set())
                        self.S_r[vtx][0] = r_tag

                    if Y_tilde_data.get((self.deg_out[vtx], r_tag, j), 0) > S1_weight.get(j, 0):
                        S1_weight[j] = Y_tilde_data.get((self.deg_out[vtx], r_tag, j), 0)
                        self.S[vtx][1][j] = Y_tilde_solution.get((self.deg_out[vtx], r_tag, j), set())
                        self.S_r[vtx][1] = r_tag

        self.vtx_S[vtx] = [S0_weight, S1_weight]

    def get_optimal_cache(self):
        for depth in sorted(list(self.depth_dict.keys()), reverse=True)[:-1]:
            for vtx in self.depth_dict[depth]:
                self.apply_on_vtx(vtx)

        self.gtc_nodes = self.get_gtc()
        print("done")

    def get_gtc(self):
        gtc_nodes = set()
        for vtx in self.S[ROOT][1][self.cache_size]:
            if self.S_r.get(vtx) and self.S_r[vtx][0] > 0:
                for gtc in set(self.policy_tree.neighbors(vtx))-self.S[ROOT][0][self.cache_size]:
                    gtc_nodes.add(gtc)
        return gtc_nodes





    @staticmethod
    def Y_data_j_to_df(Y_data, j, cache_size):
        np_data_2d = np.zeros((cache_size + 1, j + 1))
        for r in range(j + 1):
            for i in range(cache_size + 1):
                np_data_2d[i, r] = Y_data.get((j, r, i), 0)

        print("done")
        return np_data_2d

    @staticmethod
    def solution_to_df(solution, j, cache_size):
        return pd.DataFrame([[str(solution.get((j, r, i), '{}')) for r in range(j + 1)] for i in range(cache_size + 1)])

