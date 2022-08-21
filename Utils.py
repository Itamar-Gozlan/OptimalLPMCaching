import json

import networkx as nx
import ipaddress
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from collections import defaultdict, OrderedDict
import itertools
# from Algorithm import *

rule_to_vertex = None
vertex_to_rule = None
final_nodes = None
goto_nodes = []
cache_size = 4
n_bits = 32
ROOT = 0


# ---------------- Preliminaries -----------------

def binary_lpm_to_str(bin_str):
    return str(ipaddress.IPv4Address(int(bin_str + "".join((32 - len(bin_str)) * ['0']), 2))) + "/{0}".format(
        len(bin_str))


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


def generate_random_LPM(n_bits):
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


def get_cache_candidate_tree(prefix_weight_path):
    with open(prefix_weight_path, 'r') as f:
        prefix_weight = json.load(f)
    print("len(prefix_weight) = {0}".format(len(prefix_weight)))
    subtree_weight = {}
    policy_tree, rule_to_vertex, successors = HeuristicLPMCache.process_policy(prefix_weight.keys())
    vertex_to_rule = {value: key for key, value in rule_to_vertex.items()}

    # clean subtrees with weight < heaviest 1024th leaf
    cache_size = 1024
    heaviesy_cs_leaf = sorted([int(prefix_weight[vertex_to_rule[v[0]]]) for v in
                               list(filter(lambda x: x[1] == 0, {k: len(v) for k, v in successors.items()}.items()))],
                              reverse=True)[:cache_size][-1]

    depth_dict = HeuristicLPMCache.construct_depth_dict(policy_tree)
    for depth in sorted(list(depth_dict.keys()), reverse=True)[:-1]:
        for v in depth_dict[depth]:
            subtree_weight[v] = int(prefix_weight.get(vertex_to_rule[v], 0)) + \
                                sum([subtree_weight[u] for u in successors[v]])

    non_relevant_root_children = []
    for u in policy_tree.neighbors(ROOT):
        if subtree_weight[u] < heaviesy_cs_leaf:
            non_relevant_root_children.append(u)

    for u in non_relevant_root_children:
        policy_tree.remove_edge(ROOT, u)

    depth_dict = HeuristicLPMCache.construct_depth_dict(policy_tree)

    new_prefix2weight_v = list(itertools.chain(*list(depth_dict.values())))
    new_prefix2weight = {vertex_to_rule[v]: prefix_weight[vertex_to_rule[v]] for v in new_prefix2weight_v}
    new_prefix2weight[ROOT_PREFIX] = 0
    print("len(new_prefix2weight) : {0}".format(len(new_prefix2weight)))
    new_path = prefix_weight_path.replace('.json', '_filtered.json')
    print("new_path = {0}".format(new_path))
    with open(new_path, 'w') as f:
        json.dump(new_prefix2weight, f)


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_cache_candidate_tree('traces/zipf_trace_1_0_prefix2weight.json')
    get_cache_candidate_tree('traces/prefix2weight_sum60_70sorted_by_node_depth.json')
    get_cache_candidate_tree('traces/caida_traceTCP_prefix_weight.json')
    get_cache_candidate_tree('traces/caida_traceUDP_prefix_weight.json')


    # main()
