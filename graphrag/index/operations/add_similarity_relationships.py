import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from uuid import uuid4
from hashlib import sha512
from graphrag.index.utils.uuid import gen_uuid

from graphrag.index.operations.embed_text.embed_text import embed_text
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.embeddings import get_embedding_settings
from graphrag.config.models.graph_rag_config import GraphRagConfig

async def add_similarity_relationships(
    config: GraphRagConfig,
    entity_nodes: pd.DataFrame,    
    relationship_edges: pd.DataFrame,
    similarity_threshold: float = 0.9,
    callbacks=None,
    cache=None,
    embedding_field: str = "embedding"
) -> pd.DataFrame:
    """
    Augment relationship_edges with new similarity relationships computed from entity_nodes.
    Each new edge is created if the cosine similarity between the embeddings of two nodes meets or exceeds similarity_threshold.
    
    Parameters
    ----------
    entity_nodes : pd.DataFrame
        DataFrame of entities. Must contain columns 'id' and 'text'. If no embedding column exists,
        one will be computed using embed_text.
    relationship_edges : pd.DataFrame
        DataFrame containing existing relationship edges.
    similarity_threshold : float, optional
        Cosine similarity threshold between node pairs, by default 0.9.
    callbacks : WorkflowCallbacks, optional
        Callbacks instance for embed_text. Uses a no-op version if not provided.
    cache : PipelineCache, optional
        Optional cache for embed_text.
    embedding_field : str, optional
        Name of the column containing embeddings, by default "embedding".
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame containing both the existing relationship edges and the new similarity edges.
    """
    df = entity_nodes.copy()

    # Compute embeddings if not already present.
    if embedding_field not in df.columns:
        if callbacks is None:
            callbacks = NoopWorkflowCallbacks()

        text_embed_config = get_embedding_settings(config)
        embedding_strategy = text_embed_config.get("strategy", {})

        # Here we use the node "text" as the input to embedding.
        df[embedding_field] = await embed_text(
            input=df,
            callbacks=callbacks,
            cache=cache,
            embed_column="title",
            embedding_name="entity.title",
            strategy=embedding_strategy
        )
    
    # Compute cosine similarity between node embeddings.
    embeddings = np.array(df[embedding_field].tolist(), dtype=float)
    similarity_matrix = cosine_similarity(embeddings)
    
    node_titles = df["title"].tolist()
    new_edges = []        
    debug_records = []
    n = len(node_titles)
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = similarity_matrix[i, j]
            if sim_score >= similarity_threshold:

                # Let's add one more check for chemicals to see if they really are related. 
                # Chemicals with have a node name similar to this:
                # CHEM - CARBON MONOXIDE - MATRIX_OR_INT - 0.0725 NOM - 0.0775 MAX
                # or
                # CHEM - CARBON DIOXIDE - MATRIX_OR_INT - 6.0 NOM
                # Let's create a regex to extract the NOM concentration value for each node title
                # Then we'll calculate the difference between the two NOM values.
                # If greater than 40, then we don't add the edge.
                # If one of the regex doesn't match, then we'll add the edge anyway.
                if node_titles[i].startswith("CHEM -") and node_titles[j].startswith("CHEM -"):
                    import re
                    pattern = r"- ([\d\.]+) NOM"
                    match1 = re.search(pattern, node_titles[i])
                    match2 = re.search(pattern, node_titles[j])
                    if match1 and match2:
                        nom1 = float(match1.group(1))
                        nom2 = float(match2.group(1))
                        if abs(nom1 - nom2) > 40:
                            continue                
            
                normalized_weight = 5 + ((sim_score - similarity_threshold) / (1 - similarity_threshold)) * 5
                normalized_weight_int = int(round(normalized_weight))                
                new_edges.append({
                    "source": node_titles[i],
                    "target": node_titles[j],                    
                    "weight": normalized_weight_int,
                    "description": "Similarity edge",
                    "human_readable_id": None,  # will be filled below                    
                })
                debug_records.append([node_titles[i], node_titles[j], sim_score])
    similarity_edges_df = pd.DataFrame(new_edges)

    # Output debug DataFrame to an Excel file.
    if debug_records:
        debug_df = pd.DataFrame(debug_records, columns=["node_1", "node_2", "similarity_score"])
        debug_df.to_csv("debug_similar_nodes.csv", index=False)

    # Generate random hash values for id and text_unit_ids for each similarity edge.
    def generate_hash():
        return sha512(uuid4().hex.encode('utf-8'), usedforsecurity=False).hexdigest()

    similarity_edges_df["id"] = similarity_edges_df.apply(lambda row: gen_uuid(), axis=1)
    similarity_edges_df["text_unit_ids"] = similarity_edges_df.apply(lambda row: [generate_hash()], axis=1)
    
    # Combine the original relationship_edges with the new similarity edges.
    if not relationship_edges.empty:
        augmented_edges = pd.concat([relationship_edges, similarity_edges_df], ignore_index=True)

        # Reset the index and rename it to "index"
        augmented_edges = augmented_edges.reset_index(drop=False).rename(columns={"index": "index"})
        # Use the reset index as the human_readable_id (and generate an id value as well)
        augmented_edges["human_readable_id"] = augmented_edges["index"]

    else:
        augmented_edges = similarity_edges_df
    return augmented_edges
