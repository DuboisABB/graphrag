# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph, apply_clustering and run_layout methods definition."""

import logging

import networkx as nx

from graphrag.config.enums import ClusterGraphStrategyType
from graphrag.index.utils.stable_lcc import stable_largest_connected_component
from graphrag.config.models.cluster_graph_config import ClusterGraphConfig

Communities = list[tuple[int, int, int, list[str]]]


log = logging.getLogger(__name__)


def cluster_graph_according_to_config(
    graph: nx.Graph, config: ClusterGraphConfig
) -> Communities:
    """Cluster the graph based on the provided configuration."""
    if config.strategy == ClusterGraphStrategyType.app_name:
        return cluster_graph_app_name(graph)
    else:
        return cluster_graph(
            graph,
            max_cluster_size=config.max_cluster_size,
            use_lcc=config.use_lcc,
            seed=config.seed,
        )

def cluster_graph_app_name(
    graph: nx.Graph,
    application_attr: str = "APPLICATION_NAME",
) -> Communities:
    """Group nodes into communities based on their APPLICATION_NAME attribute.

    Nodes with the same value for application_attr will be in the same community.

    Parameters
    ----------
    graph : nx.Graph
        The input graph whose nodes should have an attribute defined by application_attr.
    application_attr : str, optional
        The node attribute to use for grouping communities, by default "APPLICATION_NAME".

    Returns
    -------
    Communities
        A list of tuples, each representing a community in the format
        (level, community_id, parent, list_of_node_ids). Here, level is set to 0 and parent to -1.
    """
    from collections import defaultdict

    # Group nodes by the specified application attribute.
    groups: dict[str, list[str]] = defaultdict(list)
    for node, data in graph.nodes(data=True):
        app_value = data.get(application_attr)
        if app_value is not None:
            groups[app_value].append(node)
        else:
            groups[""].append(node)  # Nodes without the attribute go to a default community

    results: Communities = []
    # Enumerate each community group, setting level 0 and parent -1.
    for community_id, (app_name, nodes) in enumerate(groups.items()):
        results.append((0, community_id, -1, nodes))

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
