from __future__ import annotations

import networkx as nx
import numpy as np


def cpm(graph: nx.Graph, partition: set, gamma: float = 1 / 7) -> float:
    """
    Calculate Constant Potts Model (CPM) for a given graph and partition.
    :param gamma: constant
    :param graph: Graph
    :param partition: Partition
    :return: The value of CPM
    """
    result = 0
    for community in partition:
        edges_community = len(nx.edges(graph, community))
        set_size = len(community)
        result += edges_community - gamma / nx.number_of_nodes(graph) * set_size
    return result


def maximize_cpm(graph: nx.Graph, partition: set, node, current_community: set) -> (float, set):
    """
    Maximize CPM by moving a node to a different partition.
    :param current_community: Actual community of the node
    :param graph: Graph
    :param partition: Partition
    :param node: Moved node
    :return: New CPM value and ccrresponding community
    """
    max_value = float("-inf")
    max_community = None

    partition.remove(current_community)
    diff_community = current_community.difference({node})
    partition.add(diff_community)

    for community in partition:
        partition_new = partition.copy()
        partition_new.remove(community)
        partition_new.add(community.union({node}))
        value = cpm(graph, partition_new)
        if value > max_value:
            max_value = value
            max_community = community
    return max_value, max_community


def move_nodes_fast(graph: nx.Graph, partition: set) -> set:
    """
    Move nodes to different partitions.
    :param graph: Graph
    :param partition: Set of partitions
    :return: New set of partitions
    """
    node_queue = list(np.random.permutation(graph.nodes()))
    while len(node_queue) > 0:
        node = node_queue.pop()
        partition_old = partition.copy()
        h_old = cpm(graph, partition_old)
        current_community = [community for community in partition_old if node in community][0]
        h_new, new_community = maximize_cpm(graph, partition_old, node, current_community)
        if h_new > h_old:
            partition.remove(new_community)
            updated_community = new_community.union({node})
            partition.add(updated_community)
            neighbors = set(nx.neighbors(graph, node)).difference(new_community)
            node_queue.extend(list(neighbors))
    return partition


def aggregate_graph(graph: nx.Graph, partition: set) -> nx.MultiGraph:
    """
    Aggregate graph.
    :param graph: Graph
    :param partition: Partition
    :return: MultiGraph
    """
    graph_new = nx.MultiGraph()
    graph_new.add_nodes_from([i for i in range(len(partition))])
    for i, community in enumerate(partition):
        for j, other_community in enumerate(partition):
            if i != j:
                edges = nx.edge_boundary(graph, community, other_community)
                graph_new.add_edges_from([(i, j) * len(edges)])
    return graph_new


def single_partition(graph: nx.Graph) -> set:
    """
    Single partition community detection algorithm.
    :param graph: Graph
    :return: Single partition (set of singletons)
    """
    return set(graph.nodes())


def merge_nodes_subset(graph: nx.Graph, partition: set, subset: set) -> set:
    """
    Merge nodes in a subset.
    :param graph: Graph
    :param partition: Partition
    :param subset: Subset of nodes
    :return: Merged partition
    """
    pass


def refine_partition(graph: nx.Graph, partition: set) -> set:
    """
    Refine partition.
    :param graph: graph
    :param partition: partition
    :return: refined partition
    """
    partition_refined = single_partition(graph)
    for community in partition:
        partition_refined = merge_nodes_subset(graph, partition_refined, community)
    return partition_refined


def leiden(graph: nx.Graph | nx.MultiGraph, partition: set = None) -> set:
    """
    Leiden community detection algorithm.
    :param graph: networkx.Graph
    :param partition: Set of partitions
    :return: Set of partitions
    """
    if partition is None:
        partition = single_partition(graph)
    done = False
    temp_graph = graph.copy()
    while not done:
        partition = move_nodes_fast(temp_graph, partition)
        done = len(partition) == temp_graph.number_of_nodes()
        if not done:
            partition_refined = refine_partition(temp_graph, partition)
            temp_graph = aggregate_graph(temp_graph, partition_refined)
            partition = single_partition(temp_graph)
    return partition
