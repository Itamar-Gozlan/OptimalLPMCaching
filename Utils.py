import json

import networkx as nx
import ipaddress
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from collections import defaultdict, OrderedDict

rule_to_vertex = None
vertex_to_rule = None
final_nodes = None
goto_nodes = []
cache_size = 4
n_bits = 32


class TrieRuleNode:
    def __init__(self, prefix, weight, depth, vertex, node_type, subtree_size=None):
        self.rule = prefix
        self.weight = weight
        self.depth = depth
        self.vertex = vertex
        self.cache = {}
        self.cache_items = {sz: set() for sz in range(cache_size + 1)}
        self.dependencies = {}
        self.node_type = node_type
        self.subtree_size = subtree_size


class RDXRuleNode:
    def __init__(self, prefix, weight, vertex):
        self.rule = prefix
        self.weight = weight
        self.vertex = vertex
        self.cache_weight = {sz: 0 for sz in range(cache_size + 1)}
        self.cache_items = {sz: set() for sz in range(cache_size + 1)}
        self.goto_nodes = {sz: set() for sz in range(cache_size + 1)}

    def __str__(self):
        global cache_size
        s = ""
        s += "v : {0} w: {1} r: {2}\n".format(self.vertex, self.weight, self.rule)
        for i in reversed(range(cache_size + 1)):
            s = s + "{0} : {1} | {2} | {3}".format(i, self.cache_weight[i], self.cache_items[i],
                                                   self.goto_nodes[i]) + "\n"
        return s

    def to_table(self):
        global cache_size
        df = pd.DataFrame({
            'sz': [cache_size - i for i in range(cache_size)],
            'weight': [self.cache_weight[cache_size - i] for i in range(cache_size)],
            'set': [self.cache_items[cache_size - i] for i in range(cache_size)],
            'goto': [self.goto_nodes[cache_size - i] for i in range(cache_size)]
        })
        return "rule: {0} weight: {1} vertex: {2}\n".format(astrix(self.rule), self.weight,
                                                            self.vertex) + df.to_markdown(index=False)


# ---------------- Preliminaries -----------------

def binary_lpm_to_str(bin_str):
    return str(ipaddress.IPv4Address(int(bin_str + "".join((32 - len(
        bin_str)) * ['0']), 2))) + "/{0}".format(len(bin_str))


def construct_tree(policy):
    T, final_nodes = build_prefix_tree(policy)
    rule_to_vertex_binary = recover(T)
    T.remove_node(-1)
    compress_prefix_tree(T, final_nodes)
    rule_to_vertex = {binary_lpm_to_str(rule): v for rule, v in rule_to_vertex_binary.items()}

    successors = {}
    for node in T.nodes:
        pred_node = list(T.predecessors(node))
        if len(pred_node) < 1:
            continue
        if len(pred_node) > 1:
            print("Error! Impossible! only one predecessor in tree")
        pred_node = pred_node[0]
        if pred_node in successors:
            successors[pred_node].append(node)
        else:
            successors[pred_node] = [node]
        successors[node] = []

    return T, rule_to_vertex, successors


def astrix(rule):
    global n_bits
    return rule if len(rule) == n_bits or rule == "*" else rule + "*"


def build_prefix_tree(paths):
    final_nodes = []

    def get_children(parent, paths):
        children = defaultdict(list)
        # Populate dictionary with key(s) as the child/children of the root and
        # value(s) as the remaining paths of the corresponding child/children
        for path in paths:
            # If path is empty, we add an edge to the NIL node.
            if not path:
                tree.add_edge(parent, NIL)
                final_nodes.append(parent)
                continue
            child, *rest = path
            # `child` may exist as the head of more than one path in `paths`.
            children[child].append(rest)
        return children

    # Initialize the prefix tree with a root node and a nil node.
    tree = nx.DiGraph()
    root = 0
    tree.add_node(root, source=None)
    NIL = -1
    tree.add_node(NIL, source="NIL")
    children = get_children(root, paths)
    stack = [(root, iter(children.items()))]
    while stack:
        parent, remaining_children = stack[-1]
        try:
            child, remaining_paths = next(remaining_children)
        # Pop item off stack if there are no remaining children
        except StopIteration:
            stack.pop()
            continue
        # We relabel each child with an unused name.
        new_name = len(tree) - 1
        # The "source" node attribute stores the original node name.
        tree.add_node(new_name, source=child)
        tree.add_edge(parent, new_name)
        children = get_children(new_name, remaining_paths)
        stack.append((new_name, iter(children.items())))

    return tree, final_nodes


def recover(T):
    NIL = -1
    root = 0
    recovered = {}
    for v in T.predecessors(NIL):
        prefix = ""
        prefix_node = v
        while v != root:
            prefix = str(T.nodes[v]["source"]) + prefix
            v = next(T.predecessors(v))  # only one predecessor
        recovered[prefix] = prefix_node
    return recovered


def compress_prefix_tree(T, final_nodes):
    final_nodes.append(0)
    for node in list(set(T.nodes) - set(final_nodes)):
        if len(list(T.predecessors(node))) > 1:
            print("Error! Node {0} has more than one predecessor")
            exit(1)

        node_pred = list(T.predecessors(node))[0]
        for descendant in list(T.successors(node)):
            T.add_edge(node_pred, descendant)
        T.remove_node(node)

    return


def generate_random_LPM():
    global n_bits
    random_ip = np.random.randint(0, 2 ** n_bits)
    n_mask_bits = np.random.randint(0, n_bits)
    # random_ip = random_ip & ~((1 << n_mask_bits) - 1)
    bit_str = "".join(['0'] * (n_bits - len(f'0b{random_ip:05b}'.split('b')[-1]))) + f'0b{random_ip:05b}'.split('b')[-1]
    return bit_str[n_mask_bits:]


def compute_random_policy(n):
    policy = set()
    while len(policy) < n:
        lpm = generate_random_LPM()
        if lpm[:-1] not in policy and lpm not in [p_lpm[:-1] for p_lpm in policy]:
            policy.add(lpm)

    return list(policy)


def compute_distance_weight():
    global rule_to_vertex
    weight = {}
    for rule, vertex in rule_to_vertex.items():
        weight[vertex] = 5 ** (n_bits - len(rule))
    return weight


# ---------------- Auxiliary -----------------

def nudge(pos, x_shift, y_shift):
    return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}


def draw_policy_trie(T, labels, figname, x_shift=0, y_shift=0):
    global final_nodes
    global goto_nodes
    global vertex_to_rule
    global cache_size

    final_nodes = vertex_to_rule[0].cache_items[cache_size]
    goto_nodes = [elm for elm in final_nodes if isinstance(elm, str)]
    goto_nodes = [int(elm.split("_")[0]) for elm in goto_nodes]
    # final_goto_nodes = [int(s.split('_')[0]) for s in
    #                     list(filter(lambda s: 'goto' in str(s), vertex_to_rule[0].cache_items[cache_size]))]
    color_map = []
    for node in T:
        # if node in final_goto_nodes:
        #     color_map.append('yellow')
        #     continue
        # if node in vertex_to_rule[0].cache_items[cache_size]:
        #     color_map.append('orange')
        #     continue
        if node in final_nodes:
            color_map.append('pink')
            continue
        if node in goto_nodes:
            color_map.append('black')
            continue
        # else:
        color_map.append('aqua')
    draw_tree(T, labels, x_shift=x_shift, y_shift=y_shift, color_map=color_map)

    plt.title(figname)
    plt.show()

    # plt.savefig('Figures/{0}.jpg'.format(figname), dpi=300)
    plt.clf()


def draw_tree(T, labels, x_shift=0, y_shift=0, color_map=None):
    pos = graphviz_layout(T, prog="dot")
    if x_shift != 0 or y_shift != 0:
        label_pos = nudge(pos, x_shift, y_shift)
        nx.draw(T, pos, node_color=color_map)
        nx.draw_networkx_labels(T, pos=label_pos, labels=labels)
    else:
        nx.draw(T, pos, node_color=color_map, with_labels=True, labels=labels)


def print_tree_different_labels(T):
    global vertex_to_rule
    # labels = {v: T.nodes[v]['source'] for v in list(T.nodes)}
    # labels[0] = "*"
    # draw_policy_trie(T, labels, 'source')
    #
    # labels = {v: vertex_to_rule[v].subtree_size for v in list(T.nodes)}
    # draw_policy_trie(T, labels, 'subtree_size')
    #
    # labels = {v: vertex_to_rule[v].weight for v in list(T.nodes)}
    # draw_policy_trie(T, labels, 'rule weight')
    #

    labels = {v: astrix(vertex_to_rule[v].rule) for v in list(T.nodes)}
    # labels = {v: str(vertex_to_rule[v]) for v in list(T.nodes)}
    #
    labels = {v: vertex_to_rule[v].to_table() for v in list(T.nodes)}
    draw_policy_trie(T, labels, 'optimal cache', x_shift=0, y_shift=-20)
    #
    # labels = {v: v for v in list(T.nodes)}
    # labels[0] = "*"
    # draw_policy_trie(T, labels, 'Nodes')

    # labels = {v: "{0} \n {1}".format(v, vertex_to_rule[v].dependencies) for v in list(T.nodes)}
    # draw_policy_trie(T, labels, 'dependencies', x_shift=0.5, y_shift=-6.5)


# ---------------- Algorithm -----------------

class TrieLPM:
    @staticmethod
    def compute_subtree_size_and_return_root_subtree_cache_set(root_subtree, descendants):
        weight = 1 if vertex_to_rule[root_subtree].node_type == 'rule' else 0
        for descendant in descendants:
            weight += vertex_to_rule[descendant].subtree_size
            if vertex_to_rule[descendant].node_type == 'rule':
                vertex_to_rule[root_subtree].dependencies[descendant] = set([descendant])
            else:
                descendant_dependncies = set()
                for d_depen in vertex_to_rule[descendant].dependencies.values():
                    descendant_dependncies = descendant_dependncies.union(d_depen)
                vertex_to_rule[root_subtree].dependencies[descendant] = descendant_dependncies
        vertex_to_rule[root_subtree].subtree_size = weight

        return set([root_subtree]) if vertex_to_rule[root_subtree].node_type == 'rule' else set()

    @staticmethod
    def compute_optimal_bottom_LPM(root_subtree, descendants, splice=True):
        """
        :param root_subtree:
        :param descendants:
        :param splice:
        :return:
        How the dependencies cope with the goto nodes? They just replace the dependencies in the current node
        """
        global rule_to_vertex
        global vertex_to_rule
        global cache_size
        global goto_nodes
        if len(descendants) > 2:
            print("Error! Not a binary tree!")
            exit(1)

        # ---- Initialization ----
        root_subtree_cache_set = TrieLPM.compute_subtree_size_and_return_root_subtree_cache_set(root_subtree,
                                                                                                descendants)

        if len(descendants) == 0:  # root_subtree is a leaf -> must be in vertex_to_rule
            vertex_to_rule[root_subtree].cache[0] = 0  # weight, size
            for i in range(1, cache_size + 1):  # 1,..,cache_size
                vertex_to_rule[root_subtree].cache[i] = vertex_to_rule[root_subtree].weight  # weight
                vertex_to_rule[root_subtree].cache_items[i] = root_subtree_cache_set
            return

        if len(descendants) == 1:
            descendant = descendants[0]
            vertex_to_rule[root_subtree].cache[0] = 0
            # ---- (a) ----
            for i in range(1, cache_size + 1):
                # ---- (a) i. ----
                if vertex_to_rule[root_subtree].subtree_size <= i:
                    vertex_to_rule[root_subtree].cache[i] = vertex_to_rule[descendant].cache[i] + vertex_to_rule[
                        root_subtree].weight
                    vertex_to_rule[root_subtree].cache_items[i] = vertex_to_rule[descendant].cache_items[i].union(
                        root_subtree_cache_set)

                # ---- (a) ii. ----
                else:
                    vertex_to_rule[root_subtree].cache[i] = vertex_to_rule[descendant].cache[i]
                    vertex_to_rule[root_subtree].cache_items[i] = vertex_to_rule[descendant].cache_items[i]

            # ---- (b) ----
            if splice and vertex_to_rule[root_subtree].node_type == 'rule':
                for i in range(1, cache_size):
                    b = 1 if vertex_to_rule[root_subtree].dependencies[descendant].issubset(
                        vertex_to_rule[descendant].cache_items[i]) else 0

                    if vertex_to_rule[root_subtree].cache[i + 1] < vertex_to_rule[descendant].cache[i - 1 + b] + \
                            vertex_to_rule[root_subtree].weight:
                        vertex_to_rule[root_subtree].cache[i + 1] = vertex_to_rule[descendant].cache[i - 1 + b] + \
                                                                    vertex_to_rule[root_subtree].weight
                        vertex_to_rule[root_subtree].cache_items[i + 1] = vertex_to_rule[descendant].cache_items[
                            i - 1 + b].union(root_subtree_cache_set)
                        if b == 0:  # mark descendant node as goto node
                            goto_nodes.append(descendant)
                            vertex_to_rule[root_subtree].cache_items[i + 1].add("{0}_goto".format(descendant))


        else:  # two descendants
            left_descendant, right_descendant = descendants[0], descendants[1]
            vertex_to_rule[root_subtree].cache[0] = 0
            # ---- (a) ----
            for i in range(1, cache_size + 1):
                cache_items = set()
                max_cache_size_i = -1
                for j in range(i + 1):
                    if max_cache_size_i < vertex_to_rule[left_descendant].cache[j] + \
                            vertex_to_rule[right_descendant].cache[i - j]:
                        max_cache_size_i = vertex_to_rule[left_descendant].cache[j] + \
                                           vertex_to_rule[right_descendant].cache[i - j]
                        cache_items = vertex_to_rule[left_descendant].cache_items[j].union(
                            vertex_to_rule[right_descendant].cache_items[i - j])
                # ---- (a) i. ----
                if vertex_to_rule[root_subtree].subtree_size <= i:
                    vertex_to_rule[root_subtree].cache[i] = max_cache_size_i + vertex_to_rule[root_subtree].weight
                    vertex_to_rule[root_subtree].cache_items[i] = cache_items.union(root_subtree_cache_set)
                # ---- (a) ii. ----
                else:
                    vertex_to_rule[root_subtree].cache[i] = max_cache_size_i
                    vertex_to_rule[root_subtree].cache_items[i] = cache_items
            # ---- (b) ----
            if splice and vertex_to_rule[root_subtree].node_type == 'rule':
                for i in range(cache_size):
                    opt_cache_val = vertex_to_rule[root_subtree].cache[i + 1]
                    curr_goto_nodes = []
                    for j in range(i):
                        # index_right_descendant = j
                        # index_left_descendant = i - j
                        legal_idx_array = TrieLPM.calculate_legal_idx(j, i - j, root_subtree, left_descendant,
                                                                      right_descendant)
                        for legal_idx in legal_idx_array:
                            left_idx, right_idx, goto_left, goto_right = legal_idx
                            curr_cache_val = vertex_to_rule[root_subtree].weight + \
                                             vertex_to_rule[left_descendant].cache[left_idx] + \
                                             vertex_to_rule[right_descendant].cache[right_idx]
                            if opt_cache_val < curr_cache_val:
                                opt_cache_val = curr_cache_val
                                new_cache_items = vertex_to_rule[left_descendant].cache_items[left_idx].union(
                                    vertex_to_rule[right_descendant].cache_items[right_idx]
                                )
                                new_cache_items.add(root_subtree)
                                for gn in [goto_left, goto_right]:
                                    if gn:
                                        new_cache_items.add(gn)
                    if opt_cache_val > vertex_to_rule[root_subtree].cache[i + 1]:
                        vertex_to_rule[root_subtree].cache[i + 1] = opt_cache_val
                        vertex_to_rule[root_subtree].cache_items[i + 1] = new_cache_items
                        # vertex_to_rule[root_subtree].dependencies = {}

    @staticmethod
    def calculate_legal_idx(idx_left, idx_right, root_subtree, left_descendant, right_descendant):
        global vertex_to_rule
        left_root_dependencies = vertex_to_rule[root_subtree].dependencies[left_descendant]
        right_root_dependencies = vertex_to_rule[root_subtree].dependencies[right_descendant]
        idx_left_in = True if left_root_dependencies.issubset(
            vertex_to_rule[left_descendant].cache_items[idx_left]) else False
        idx_right_in = True if right_root_dependencies.issubset(
            vertex_to_rule[right_descendant].cache_items[idx_right]) else False

        if idx_left > 0 and left_root_dependencies.issubset(vertex_to_rule[left_descendant].cache_items[idx_left - 1]):
            idx_left_m1_in = True
        else:
            idx_left_m1_in = False

        if idx_right > 0 and right_root_dependencies.issubset(
                vertex_to_rule[right_descendant].cache_items[idx_right - 1]):
            idx_right_m1_in = True
        else:
            idx_right_m1_in = False

        legal_idx = []
        if idx_left_in and idx_right_in:
            legal_idx.append((idx_left, idx_right, None, None))  # no additional goto nodes if possible
        if idx_left >= 2:
            legal_idx.append((idx_left - 2, idx_right, "{0}_goto".format(left_descendant),
                              "{0}_goto".format(right_descendant)))  # reduce 2 goto nodes from left
            if idx_left_m1_in:
                legal_idx.append((idx_left - 1, idx_right, None,
                                  "{0}_goto".format(right_descendant)))  # reduce 1 goto nodes from left if possible
        if idx_right >= 2:
            legal_idx.append((idx_left, idx_right - 2, "{0}_goto".format(left_descendant),
                              "{0}_goto".format(right_descendant)))  # reduce 2 goto nodes from right
            if idx_right_m1_in:
                legal_idx.append((idx_left, idx_right - 1, "{0}_goto".format(left_descendant),
                                  None))  # reduce 1 goto nodes from right if possible

        if idx_right >= 1 and idx_left >= 1:
            legal_idx.append((idx_left - 1, idx_right - 1, "{0}_goto".format(left_descendant),
                              "{0}_goto".format(right_descendant)))  # reduce 1 goto nodes from each

        # Each tuple should preserve 'new_left_idx'+'new_right_idx' + |goto_nodes| = left_idx + right_idx

        return list(set(legal_idx))


class RadixLPM:
    @staticmethod
    def compute_optimal_cache_two_descendants(left, right):
        global cache_size
        next_node = RDXRuleNode(prefix=None, weight=None, vertex=None)  # Using node just to save intermediate values
        for i in range(1, cache_size + 1):
            cache_max_val = 0
            cache_best_items = set()
            for j in range(i + 1):
                if left.cache_weight[j] + right.cache_weight[i - j] > cache_max_val:
                    cache_best_items = left.cache_items[j].union(right.cache_items[i - j])
                    cache_max_val = left.cache_weight[j] + right.cache_weight[i - j]
            next_node.cache_weight[i] = cache_max_val
            next_node.cache_items[i] = cache_best_items
        return next_node

    @staticmethod
    def compute_optimal_cache_among_descendants(descendant_array, output_graph=None):
        if len(descendant_array) == 1:
            return descendant_array[0]

        # if len(descendant_array) == 2:
        #     return RadixLPM.compute_optimal_cache_two_descendants(descendant_array[0], descendant_array[1])

        next_level_descendant_array = []
        count = 1
        for i in range(0, len(descendant_array) - 1, 2):
            next_level_node = RadixLPM.compute_optimal_cache_two_descendants(descendant_array[i],
                                                                             descendant_array[i + 1])
            next_level_descendant_array.append(next_level_node)
            count += 1
            if output_graph is not None:
                next_level_node_id = max([node.vertex for node in descendant_array]) + count
                next_level_node.vertex = next_level_node_id
                output_graph.add_node(next_level_node_id, RDXRuleNode=next_level_node)
                output_graph.add_node(descendant_array[i].vertex, RDXRuleNode=descendant_array[i])
                output_graph.add_node(descendant_array[i + 1].vertex, RDXRuleNode=descendant_array[i + 1])
                print("next_level_node: {0}".format(next_level_node.vertex))
                output_graph.add_edge(next_level_node.vertex, descendant_array[i].vertex)
                output_graph.add_edge(next_level_node.vertex, descendant_array[i + 1].vertex)
                print("connecting {0}->{1}, {0}->{2}".format(next_level_node.vertex, descendant_array[i].vertex,
                                                             descendant_array[i + 1].vertex))

        if len(descendant_array) % 2 != 0:  # odd number of descendants -> add last one to the next round
            next_level_descendant_array.append(descendant_array[-1])
            if output_graph is not None:
                print("next_level_descendant_array.append({0})".format(descendant_array[-1].vertex))

        return RadixLPM.compute_optimal_cache_among_descendants(next_level_descendant_array, output_graph)

    @staticmethod
    def compute_optimal_cache_for_depth(T, predecessors):
        global vertex_to_rule
        global cache_size
        for v in predecessors:
            if T.out_degree(v) == 0:  # v is a leaf
                for i in range(1, cache_size + 1):
                    vertex_to_rule[v].cache_items[i] = set([v])
                    vertex_to_rule[v].cache_weight[i] = vertex_to_rule[v].weight
            else:
                result = RadixLPM.compute_optimal_cache_among_descendants([vertex_to_rule[u] for u in T.successors(v)])
                vertex_to_rule[v].cache_items = result.cache_items
                vertex_to_rule[v].cache_weight = result.cache_weight
                for i in range(1, cache_size + 1):
                    vertex_to_rule[v].goto_nodes[i] = set(T.successors(v)) - result.cache_items[i]

                for i in range(1, cache_size + 1):
                    if len(vertex_to_rule[v].goto_nodes[i]) == 0 and len(
                            vertex_to_rule[v].cache_items[i]) < i:  # all dependencies satisfy
                        vertex_to_rule[v].cache_items[i].add(v)
                        vertex_to_rule[v].cache_weight[i] += vertex_to_rule[v].weight

                # for i in range(1, cache_size + 1):
                #     j = i + len(vertex_to_rule[v].goto_nodes[i])
                #     if j + 1 <= cache_size:
                #         if v in vertex_to_rule[v].cache_items[j + 1]:
                #             continue
                #         if vertex_to_rule[v].cache_weight[j + 1] < vertex_to_rule[v].cache_weight[i] + vertex_to_rule[
                #             v].weight:
                #             vertex_to_rule[v].cache_weight[j + 1] = vertex_to_rule[v].cache_weight[i] + vertex_to_rule[
                #                 v].weight
                #             vertex_to_rule[v].cache_items[j + 1] = vertex_to_rule[v].cache_items[i].union(
                #                 map(lambda vtg : str(vtg) + "_goto",
                #                 vertex_to_rule[v].goto_nodes[i]))
                #             vertex_to_rule[v].cache_items[j + 1].add(v)

    @staticmethod
    def create_binary_scheme(arr, output_graph):
        if len(arr) == 1:
            return

        next_level_array = []
        count = 1
        for i in range(0, len(arr) - 1, 2):
            next_level_node = max(arr) + count
            count += 1
            print("next_level_node: {0}".format(next_level_node))
            output_graph.add_edge(next_level_node, arr[i])
            output_graph.add_edge(next_level_node, arr[i + 1])
            print("connecting {0}->{1}, {0}->{2}".format(next_level_node, arr[i], arr[i + 1]))
            next_level_array.append(next_level_node)

        if len(arr) % 2 != 0:
            next_level_array.append(arr[-1])
            print("next_level_array.append({0})".format(arr[-1]))

        RadixLPM.create_binary_scheme(next_level_array, output_graph)

    @staticmethod
    def create_binary_scheme_recursion(arr, output_graph):
        if len(arr) == 1:
            return arr[0]

        n = len(arr)
        node_a = RadixLPM.create_binary_scheme_recursion(arr[:int(n / 2)], output_graph)
        node_b = RadixLPM.create_binary_scheme_recursion(arr[n - int(n / 2):], output_graph)

        new_node = 5 ** 3 + node_a
        output_graph.add_edge(new_node, node_a)
        output_graph.add_edge(new_node, node_b)

        return new_node


def compute_TrieLPM(T, weight):
    global rule_to_vertex
    global vertex_to_rule
    global final_nodes
    global goto_nodes
    global n_bits
    vertex_to_rule = {vertex: TrieRuleNode(rule, weight[vertex], -1, vertex, 'rule') for rule, vertex in
                      rule_to_vertex.items()}
    for node in T.nodes:
        if node in vertex_to_rule:
            continue
        else:
            vertex_to_rule[node] = TrieRuleNode(None, 0, -1, node, None)

    root = 0
    T.remove_node(-1)
    shortest_path = nx.single_source_shortest_path(T, root)
    depth_dict = {}
    for v in T.nodes:
        d = len(shortest_path[v])
        depth_dict[d] = [v] + depth_dict.get(d, [])

    for depth in sorted(list(depth_dict.keys()), reverse=True):
        for root_subtree in depth_dict[depth]:
            TrieLPM.compute_optimal_bottom_LPM(root_subtree, list(T.successors(root_subtree)))

    print_tree_different_labels(T)


def playground():
    with open("/home/itamar/PycharmProjects/CacheSimulator/DCTraces/caida_policy.json", 'r') as f:
        policy = json.load(f)

    T, final_nodes = build_prefix_tree(list(policy.values()))
    rule_to_vertex = recover(T)
    T.remove_node(-1)
    compress_prefix_tree(T, final_nodes)
    print("s")
    pos = graphviz_layout(T, prog="dot")
    nx.draw(T, pos)
    plt.show()
    return T, final_nodes


def main():
    global cache_size
    global rule_to_vertex
    global vertex_to_rule
    T, final_nodes = construct_tree()

    weight = compute_distance_weight()
    vertex_to_rule = {vertex: RDXRuleNode(vertex=vertex, weight=weight[vertex], prefix=rule) for
                      rule, vertex in rule_to_vertex.items()}
    root = 0
    vertex_to_rule[root] = RDXRuleNode(vertex=root, weight=0, prefix="*")

    shortest_path = nx.single_source_shortest_path(T, root)
    depth_dict = {}
    for v in T.nodes:
        d = len(shortest_path[v])
        depth_dict[d] = [v] + depth_dict.get(d, [])

    for depth in sorted(list(depth_dict.keys()), reverse=True):
        RadixLPM.compute_optimal_cache_for_depth(T, depth_dict[depth])

    print_tree_different_labels(T)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    playground()
    # main()
