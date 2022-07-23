import abc
from Utils import construct_tree
import ipaddress
import networkx as nx
import itertools
import copy

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


class Algorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_cache(self, prefix_weight):
        pass


class OptimalLPMCache(Algorithm):
    def __init__(self, cache_size: int, policy: list, dependency_splice=True):
        self.policy_tree, self.rule_to_vertex, self.successors = OptimalLPMCache.process_policy(policy)
        self.depth_dict = OptimalLPMCache.construct_depth_dict(self.policy_tree)
        self.vertex_to_rule = {value: key for key, value in self.rule_to_vertex.items()}
        self.cache_size = cache_size
        self.dependency_splice = dependency_splice
        self.feasible_set = {}

    def get_cache(self, prefix_weight):
        cache_candidate_depth_dict = self.get_cache_candidate_tree(prefix_weight)
        print("SUM: {0}".format(sum([len(l) for l in cache_candidate_depth_dict.values()])))
        for depth in sorted(list(cache_candidate_depth_dict.keys()), reverse=True):
            for v in cache_candidate_depth_dict[depth]:
                if len(self.successors[v]) == 0:  # leaf
                    continue
                else:
                    children_feasible_set = FeasibleSet.OptDTUnion(self.get_candidate_feasible_set(v, prefix_weight),
                                                                   self.cache_size)
                    v_in_cache = {i: False for i in range(self.cache_size + 1)}
                    # Adding v
                    for i in range(self.cache_size + 1):
                        if children_feasible_set.item_count[i] < i and children_feasible_set.superset_of(i,
                                                                                                         self.successors[
                                                                                                             v]):
                            children_feasible_set.insert_iset_item(i, v, prefix_weight.get(self.vertex_to_rule[v], 0))
                            v_in_cache[i] = True

                    if self.dependency_splice:
                        self.apply_dependency_splice(v, prefix_weight, children_feasible_set, v_in_cache)
                    else:
                        self.feasible_set[v] = children_feasible_set
        if ADD_GOTO_NODES:
            get_rule = lambda vtx: self.vertex_to_rule[vtx] if isinstance(vtx, int) else self.vertex_to_rule[int(
                vtx.split("_")[0])] + "_goto"
        else:
            get_rule = lambda vtx: self.vertex_to_rule[vtx]
        return set([get_rule(v) for v in self.feasible_set[ROOT].feasible_iset[self.cache_size]])

    def get_candidate_feasible_set(self, v, prefix_weight):
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
        for i in range(self.cache_size + 1):
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

    def apply_dependency_splice(self, v, prefix_weight, children_feasible_set, v_in_cache):
        v_weight = prefix_weight.get(self.vertex_to_rule[v], 0)
        dependency_spliced_v = FeasibleSet()
        spliced = False
        for i in range(self.cache_size + 1):
            j_maybe = i + len(
                set(self.policy_tree.successors(v)) - children_feasible_set.feasible_iset[i])
            if j_maybe + 1 <= self.cache_size and not v_in_cache[j_maybe + 1]:
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
            for i in range(self.cache_size + 1):
                if dependency_spliced_v.item_count.get(i) is None:  # copy
                    dependency_spliced_v.item_count[i] = children_feasible_set.item_count[i]
                    dependency_spliced_v.feasible_iset[i] = children_feasible_set.feasible_iset[i]
                    dependency_spliced_v.feasible_iset_weight[i] = \
                        children_feasible_set.feasible_iset_weight[i]

            self.feasible_set[v] = dependency_spliced_v
        else:
            self.feasible_set[v] = children_feasible_set

    def OptDTUnion(self, feasible_set, children, prefix_weight):
        if len(children) == 1:
            return feasible_set[children[0]]
        split_i = int(len(children) / 2)
        feasible_set_x = self.OptDTUnion(feasible_set, children[:split_i], prefix_weight)
        feasible_set_y = self.OptDTUnion(feasible_set, children[split_i:], prefix_weight)
        feasible_set = {}
        for i in range(self.cache_size + 1):
            max_i = set()
            for j in range(i + 1):
                max_i = max(max_i, feasible_set_x[j].union(feasible_set_y[i - j]),
                            key=lambda S: OptimalLPMCache.set_weight(S, prefix_weight))
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

        depth_dict = OptimalLPMCache.construct_depth_dict(self.policy_tree)

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

        return construct_tree(binary_policy)

    @staticmethod
    def set_weight(S, prefix_weight):
        return sum([prefix_weight.get(elem, 0) for elem in S])


class PowerOfKChoices(Algorithm):
    def __init__(self, cache_size):
        Algorithm.__init__(self)
        self.cache_size = cache_size

    def get_cache(self, prefix_weight, ):
        return set((map(lambda v: v[0], sorted(list(prefix_weight.items()),
                                               key=lambda v: v[1], reverse=True)[:self.cache_size])))
