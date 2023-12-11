from __future__ import annotations

import networkx as nx
import numpy as np
from copy import deepcopy


gamma: float = 1 / 7
theta: float = 0.5


def cpm(graph: nx.Graph, partition: list) -> float:
    """
    Calculate Constant Potts Model (CPM) for a given graph and partition.
    :param graph: Graph
    :param partition: Partition
    :return: The value of CPM
    """
    result = 0
    for community in partition:
        edges_community = nx.subgraph(graph, community).number_of_edges()
        set_size = len(community)
        result += edges_community - gamma / nx.number_of_nodes(graph) * set_size
    return result


def maximize_cpm(graph: nx.Graph, partition: list, node, current_community: set) -> (float, set):
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
    if len(diff_community) > 0:
        partition.append(diff_community)

    for community in partition:
        partition_new = deepcopy(partition)
        community_new = community.copy()
        partition_new.remove(community_new)
        partition_new.append(community_new.union({node}))
        value = cpm(graph, partition_new)
        if value > max_value:
            max_value = value
            max_community = community

    return max_value, max_community


def delta_cpm(graph: nx.Graph, partition: list, node: str, community: set) -> (float, set):
    """
    Calculate delta CPM.
    :param graph: Graph
    :param partition: Partition
    :param node: Node
    :param community: Community
    :return: Delta CPM
    """
    partition_new = deepcopy(partition)
    partition_new.remove(community)
    community_new = community.union({node})
    partition_new.append(community_new)
    return cpm(graph, partition_new) - cpm(graph, partition), community_new


def move_nodes_fast(graph: nx.Graph, partition: list) -> list:
    """
    Move nodes to different partitions.
    :param graph: Graph
    :param partition: Set of partitions
    :return: New set of partitions
    """
    node_queue = list(np.random.permutation(graph.nodes()))
    while len(node_queue) > 0:
        node = node_queue.pop()
        partition_old = deepcopy(partition)
        h_old = cpm(graph, partition_old)
        current_community = [community for community in partition_old if node in community][0]
        h_new, new_community = maximize_cpm(graph, partition_old, node, current_community)
        if h_new > h_old:
            partition.remove(new_community)
            community = [community for community in partition if node in community][0]
            if len(community) == 1:
                partition.remove(community)
            else:
                community.remove(node)
            updated_community = new_community.union({node})
            partition.append(updated_community)
            neighbors = set(nx.neighbors(graph, node)).difference(new_community)
            neighbors = neighbors.difference(set(node_queue))
            node_queue.extend(list(neighbors))
    return partition


def aggregate_graph(graph: nx.Graph, partition: list) -> nx.MultiGraph:
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
            if i <= j:
                edges = list(nx.edge_boundary(graph, community, other_community))
                graph_new.add_edges_from([(i, j)] * len(edges))
    return graph_new


def single_partition(graph: nx.Graph) -> list:
    """
    Single partition community detection algorithm.
    :param graph: Graph
    :return: Single partition (set of singletons)
    """
    return [{elem} for elem in nx.nodes(graph)]


def merge_nodes_subset(graph: nx.Graph, partition: list, subset: set) -> list:
    """
    Merge nodes in a subset.
    :param graph: Graph
    :param partition: Partition
    :param subset: Subset of nodes
    :return: Merged partition
    """

    r = {node for node in subset if len(nx.subgraph(graph, subset).edges(node)) >= gamma * (len(subset) - 1)}
    for node in r:
        community = [community for community in partition if node in community][0]
        if len(community) == 1:
            partition.remove(community)
            v = community.pop()
            t = [c for c in partition if c.issubset(subset) and len(list(nx.edge_boundary(graph, c, subset.difference(c)))) >= gamma * len(c) * (len(subset) - len(c))]
            if len(t) > 0:
                distribution = [np.exp(delta_cpm(graph, partition, v, c)[0] / theta) for c in t]
                distribution = [elem if elem >= 0 else 0 for elem in distribution]
                distribution = [elem / sum(distribution) for elem in distribution]
                community = np.random.choice(t, p=distribution)
                partition.remove(community)
                community = community.union({node})
                partition.append(community)
    return partition


def refine_partition(graph: nx.Graph, partition: list) -> list:
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


def leiden(graph: nx.Graph | nx.MultiGraph, partition: list = None) -> list:
    """
    Leiden community detection algorithm.
    :param graph: networkx.Graph
    :param partition: Set of partitions
    :return: Set of partitions
    """
    if not graph.is_multigraph():
        graph = nx.MultiGraph(graph)
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


if __name__ == '__main__':
    g = nx.complete_graph(8)
    g = nx.union(g, nx.complete_graph(8), ('a-', 'b-'))
    g.add_edge('a-0', 'b-0')
    print(leiden(g))
