# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Custom cluster graph functions for multiwave"""

import logging
import pprint
import networkx as nx
import threading
import asyncio
import re
from sklearn.metrics.pairwise import cosine_similarity

from graphrag.config.enums import ClusterGraphStrategyType
from graphrag.config.embeddings import get_embedding_settings
from graphrag.config.models.graph_rag_config import GraphRagConfig

from graphrag.index.utils.stable_lcc import stable_largest_connected_component
from graphrag.config.models.cluster_graph_config import ClusterGraphConfig
from collections import defaultdict



import numpy as np
import pandas as pd
from graphrag.index.operations.embed_text.embed_text import embed_text
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.index.operations.cluster_graph import cluster_graph, _compute_leiden_communities

Communities = list[tuple[int, int, int, list[str]]]

log = logging.getLogger(__name__)

def run_async_in_thread(coro, *args, **kwargs):
    result = None

    def target():
        nonlocal result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro(*args, **kwargs))
        loop.close()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return result

async def cluster_graph_embedding(
    nodes: pd.DataFrame,   # Instance of PipelineCache (optional)
    strategy: dict,
    cluster_config: ClusterGraphConfig,
    similarity_threshold: float = 0.9,
    callbacks: WorkflowCallbacks=None,  # Instance of WorkflowCallbacks (optional)
    cache: PipelineCache=None,       
) -> list[tuple[int, int, int, list[str]]]:
    """
    Cluster nodes based on the similarity of their text embeddings.

    The approach is:
      1. Use embed_text (existing functionality) to obtain embeddings for each node.
      2. Compute pairwise cosine similarity among embeddings.
      3. Create edges between nodes if their similarity exceeds similarity_threshold.
      4. Extract connected components (each component is a community).

    Parameters
    ----------
    nodes : pd.DataFrame
        A DataFrame containing node data with at least columns "id" and "text".
    callbacks : WorkflowCallbacks
        Callbacks object from the pipeline.
    cache : PipelineCache
        Pipeline cache instance.
    strategy : dict
        Embedding strategy configuration (see embed_text.py for details).
    similarity_threshold : float, optional
        Minimum cosine similarity required to link two nodes, by default 0.8.

    Returns
    -------
    list[tuple[int, int, int, list[str]]]
        A list of communities, each represented as a tuple:
            (level, community_id, parent, list_of_node_ids).
    """
    # Copy to avoid modifying the input DataFrame
    df = nodes.copy()

    if callbacks is None:
        callbacks = NoopWorkflowCallbacks()

    # Get embeddings for the nodes using embed_text.
    # This call leverages the existing embed_text functionality.
    df["embedding"] = await embed_text(
        input=df,
        callbacks=callbacks,
        cache=cache,
        embed_column="text",
        embedding_name="entity.title",
        strategy=strategy,
    )

    # Build a similarity graph based on cosine similarity.
    G = nx.Graph()
    for _, row in df.iterrows():
        node_id = row["id"]
        G.add_node(node_id)

    embeddings_list = df["embedding"].tolist()
    node_ids = df["id"].tolist()

    # Convert embeddings to a numpy array and normalize them.
    embeddings_np = np.array(embeddings_list, dtype=float)
    similarity_matrix = cosine_similarity(embeddings_np)

    num_nodes = len(node_ids)
    
    # Prepare list to collect similar node pairs for debugging.
    debug_records = []

    # Compare each pair of nodes.
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            score = similarity_matrix[i, j]
            if score >= similarity_threshold:
                G.add_edge(node_ids[i], node_ids[j])
                debug_records.append([node_ids[i], node_ids[j], score])

    # Output debug DataFrame to an Excel file.
    if debug_records:
        debug_df = pd.DataFrame(debug_records, columns=["node_id_1", "node_id_2", "similarity_score"])
        debug_df.to_csv("debug_similar_nodes.csv", index=False)
        
    # Extract communities as connected components.
    if False:
        connected_components = list(nx.connected_components(G))
        communities = []
        for community_id, comp in enumerate(connected_components):
            communities.append((0, community_id, -1, list(comp)))
    else:
        # Compute communities using the Leiden algorithm.
        # _compute_leiden_communities returns a tuple of (node_id_to_community_map, parent_mapping)
        node_id_to_community_map, parent_mapping = _compute_leiden_communities(
            G,
            max_cluster_size=cluster_config.max_cluster_size,
            use_lcc=cluster_config.use_lcc,
            seed=cluster_config.seed
        )
        levels = sorted(node_id_to_community_map.keys())
        communities = []
        for level in levels:
            clusters = {}
            for node_id, community_id in node_id_to_community_map[level].items():
                clusters.setdefault(community_id, []).append(node_id)
            for community_id, nodes in clusters.items():
                communities.append((level, community_id, parent_mapping[community_id], nodes))
    
    return communities


def cluster_graph_according_to_config(
    graph: nx.Graph, 
    config: GraphRagConfig,
    callbacks: WorkflowCallbacks=None,  # Instance of WorkflowCallbacks (optional)
    cache: PipelineCache=None
) -> Communities:
    """Cluster the graph based on the provided configuration.
    
    This method runs both community clustering strategies and writes their results
    to separate debug files. However, only the community result that matches the config's
    strategy is returned.
    """

        # Extract the cluster_graph config from the overall configuration
    cluster_config = config.cluster_graph
    # Run both strategies
    communities_app = run_async_in_thread(cluster_graph_app_name, graph, config, callbacks, cache)
    communities_general = cluster_graph(
        graph,
        max_cluster_size=cluster_config.max_cluster_size,
        use_lcc=cluster_config.use_lcc,
        seed=cluster_config.seed,
    )
    
    # Write both communities to debug files
    with open("debug_app_name_communities.txt", "w") as f:
        f.write("Cluster Graph - APP Name Strategy:\n")
        f.write(pprint.pformat(communities_app))
    
    with open("debug_general_communities.txt", "w") as f:
        f.write("Cluster Graph - General Strategy:\n")
        f.write(pprint.pformat(communities_general))
    
    # Return communities based on config strategy
    if cluster_config.strategy == ClusterGraphStrategyType.app_name:
        return communities_app
    else:
        return communities_general

async def cluster_graph_app_name(
    graph: nx.Graph,
    config: GraphRagConfig,
    callbacks: WorkflowCallbacks=None,  # Instance of WorkflowCallbacks (optional)
    cache: PipelineCache=None
    ) -> Communities:
    """
    Group nodes into communities based on additional conditions:
    
    - If the node's name starts with "CHEM", use a regex pattern to extract a community name.
      The pattern matches something like: "CHEM - <any character except dash> - <any character except dash>".
    - All other nodes are grouped using an embedding-based clustering.
    
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
            if not added_to_app:
                other_nodes.append(node_name)
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

    # For the remaining nodes, use embedding-based clustering.
    if other_nodes:
        # Create a DataFrame for the other nodes.
        # Here we simply use the node name as the text to embed.
        df_nodes = pd.DataFrame({"id": other_nodes, "text": other_nodes})

        if callbacks is None:
            callbacks = NoopWorkflowCallbacks()

        # Retrieve embedding strategy.
        cluster_config = config.cluster_graph
        text_embed_config = get_embedding_settings(config)
        embedding_strategy = text_embed_config.get("strategy", {})
        similarity_threshold = cluster_config.similarity_threshold

        communities_embedding = await cluster_graph_embedding(
            nodes=df_nodes,            
            strategy=embedding_strategy,
            cluster_config=cluster_config,
            similarity_threshold=similarity_threshold,
            callbacks=callbacks,
            cache=cache
        )
        for level, cid, parent, comp_nodes in communities_embedding:
            results.append((level, community_id_counter, parent, comp_nodes))
            community_id_counter += 1
    
    return results