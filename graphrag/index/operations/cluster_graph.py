# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph, apply_clustering and run_layout methods definition."""

import logging
import pprint
import networkx as nx
import re

from graphrag.config.enums import ClusterGraphStrategyType
from graphrag.index.utils.stable_lcc import stable_largest_connected_component
from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from collections import defaultdict

Communities = list[tuple[int, int, int, list[str]]]


log = logging.getLogger(__name__)

def cluster_graph_according_to_config(
    graph: nx.Graph, config: ClusterGraphConfig
) -> Communities:
    """Cluster the graph based on the provided configuration.
    
    This method runs both community clustering strategies and writes their results
    to separate debug files. However, only the community result that matches the config's
    strategy is returned.
    """
    # Run both strategies
    communities_app = cluster_graph_app_name(graph)
    communities_general = cluster_graph(
        graph,
        max_cluster_size=config.max_cluster_size,
        use_lcc=config.use_lcc,
        seed=config.seed,
    )
    
    # Write both communities to debug files
    with open("debug_app_name_communities.txt", "w") as f:
        f.write("Cluster Graph - APP Name Strategy:\n")
        f.write(pprint.pformat(communities_app))
    
    with open("debug_general_communities.txt", "w") as f:
        f.write("Cluster Graph - General Strategy:\n")
        f.write(pprint.pformat(communities_general))
    
    # Return communities based on config strategy
    if config.strategy == ClusterGraphStrategyType.app_name:
        return communities_app
    else:
        return communities_general

def cluster_graph_app_name(graph: nx.Graph) -> Communities:
    """
    Group nodes into communities based on additional conditions:
    
    - If the node's name starts with "CHEM", use a regex pattern to extract a community name.
      The pattern matches something like: "CHEM - <any character except dash> - <any character except dash>".
    - All other nodes are grouped using the Leiden clustering algorithm.
    
    Returns
    -------
    Communities
        A list of tuples in the format (level, community_id, parent, list_of_node_ids).
    """
    chem_groups: dict[str, list[str]] = defaultdict(list)
    app_groups: dict[str, list[str]] = defaultdict(list)
    other_nodes: list[str] = []

    # Partition nodes into CHEM nodes (with regex grouping) and the rest.
    for node in graph.nodes:
        node_name = str(node)
        if node_name.startswith("CHEM"):
            # Using a regex to match: "CHEM - <anything except dash> - <anything except dash>"
            m = re.match(r"^(CHEM\s*-\s*[^-]+\s*-\s*[^-]+)", node_name)
            if m:
                group_key = m.group(1)
            else:
                group_key = "CHEM"
            chem_groups[group_key].append(node_name)
        elif node_name.startswith("APP"):
            group_key = node_name
            app_groups[group_key].append(node_name)
        elif node_name.startswith("QUOTE"):
            # Check if this quote node is connected to any APP node.
            added_to_app = False
            for neighbor in graph.neighbors(node):
                neighbor_name = str(neighbor)
                if neighbor_name.startswith("APP"):
                    # Add the quote node to the community of the found APP node.
                    app_groups[neighbor_name].append(node_name)
                    added_to_app = True
                    break            
        else:
            other_nodes.append(node_name)

    results: Communities = []
    community_id_counter = 0

    # Create communities for nodes that start with CHEM
    for group_key, nodes in chem_groups.items():
        results.append((0, community_id_counter, -1, nodes))
        community_id_counter += 1

    # Create communities for APP nodes (including attached QUOTE nodes)
    for group_key, nodes in app_groups.items():
        results.append((0, community_id_counter, -1, nodes))
        community_id_counter += 1        

    # For the remaining nodes, run the Leiden clustering algorithm.
    if other_nodes:
        subgraph = graph.subgraph(other_nodes)
        # Handle subgraphs without edges, which cause hierarchical_leiden to raise EmptyNetworkError.
        if subgraph.number_of_edges() == 0:
            for node in subgraph.nodes:
                results.append((0, community_id_counter, -1, [node]))
                community_id_counter += 1
        else:
            leiden_communities = cluster_graph(
                subgraph, max_cluster_size=1000, use_lcc=False, seed=None
            )
            for level, _, parent, nodes in leiden_communities:
                results.append((level, community_id_counter, parent, nodes))
                community_id_counter += 1

    return results


def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> Communities:
    """Apply a hierarchical clustering algorithm to a graph."""
    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes")
        return []

    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )

    levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result = {}
        clusters[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = raw_community_id
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping[cluster_id], nodes))
    return results


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    # NOTE: This import is done here to reduce the initial import time of the graphrag package
    from graspologic.partition import hierarchical_leiden

    if use_lcc:
        graph = stable_largest_connected_component(graph)

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return results, hierarchy
