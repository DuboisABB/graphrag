# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to create the base entity graph."""

import pandas as pd

from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.cluster_graph_mw import cluster_graph_according_to_config
from graphrag.index.operations.create_graph import create_graph


def compute_communities(
    base_relationship_edges: pd.DataFrame,
    config: GraphRagConfig
) -> pd.DataFrame:
    """All the steps to create the base entity graph."""
    graph = create_graph(base_relationship_edges)

    #communities = cluster_graph(
    #    graph,
    #    max_cluster_size,
    #    use_lcc,
    #    seed=seed,
    #)

    communities = cluster_graph_according_to_config(graph, config)
    #print("communities:", communities, type(communities))

    base_communities = pd.DataFrame(
        communities, columns=pd.Index(["level", "community", "parent", "title"])
    ).explode("title")
    base_communities["community"] = base_communities["community"].astype(int)

    return base_communities
