import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from graphrag.config.embeddings import get_embedding_settings
from graphrag.config.models.graph_rag_config import GraphRagConfig

# We need to call embed_text from our embeddings module.
from graphrag.index.operations.embed_text import embed_text

# Provide a default callback if none is given.
try:
    from graphrag.callbacks.workflow_callbacks import NoopWorkflowCallbacks
except ImportError:
    # Fallback in case NoopWorkflowCallbacks isn't available; you may replace with your own dummy.
    class NoopWorkflowCallbacks:
        def progress(self, *args, **kwargs):
            pass

async def normalize_entities(
    entities_df: pd.DataFrame,
    text_embed_config: dict,
    callbacks=None,  # Instance of WorkflowCallbacks (optional)
    cache=None,      # Instance of PipelineCache (optional)
    threshold: float = 0.85
) -> pd.DataFrame:
    """
    Normalize entity names by computing embeddings for each entity title using the embedding strategy
    from the GraphRag configuration. Similar titles (based on cosine similarity of their embeddings)
    are then merged under a canonical normalized title.
    
    This implementation reuses the embedding mechanism as generate_text_embeddings by calling embed_text().

    Parameters
    ----------
    entities_df : pd.DataFrame
        DataFrame with extracted entities. It must have a column 'title'.
    config : GraphRagConfig
        The GraphRag configuration object used to obtain embedding settings.
    callbacks
        WorkflowCallbacks instance. If None, a NoopWorkflowCallbacks is used.
    cache
        PipelineCache instance. If None, defaults to None.
    threshold : float, optional
        Cosine similarity threshold above which two entity titles are considered identical.
    
    Returns
    -------
    pd.DataFrame
        A copy of entities_df with an additional column 'title_normalized'.
    """
    if 'title' not in entities_df.columns:
        raise ValueError("The DataFrame must contain a 'title' column.")

    if callbacks is None:
        callbacks = NoopWorkflowCallbacks()
    # cache remains None if not provided; update as needed if you have a default PipelineCache implementation

    texts = entities_df['title'].tolist()

    # Retrieve embedding settings from config.
    strategy = text_embed_config.get("strategy", {})

    # Prepare a temporary DataFrame with 'id' and 'title' columns as required by embed_text().
    tmp_df = pd.DataFrame({
        "id": list(range(len(texts))),
        "title": texts
    })

    # Use embed_text function to compute embeddings.
    result_embeddings = await embed_text(
        input=tmp_df,
        callbacks=callbacks,
        cache=cache,
        embed_column="title",
        strategy=strategy,
        embedding_name="entity.title",
        id_column="id",
        title_column="title"
    )
    embeddings = np.array(result_embeddings)

    # Compute cosine similarity matrix between embeddings.
    sim_matrix = cosine_similarity(embeddings)
    
    # Group similar entities based on the threshold.
    n = len(texts)
    labels = [-1] * n
    next_label = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = next_label
        for j in range(i + 1, n):
            if labels[j] == -1 and sim_matrix[i, j] >= threshold:
                labels[j] = next_label
        next_label += 1

    # For each group, choose a canonical title (for example, the first encountered title).
    label_to_canonical = {}
    for label, text in zip(labels, texts):
        if label not in label_to_canonical:
            label_to_canonical[label] = text

    normalized_titles = [label_to_canonical[label] for label in labels]
    
    # Create a debug dataframe to assist with troubleshooting
    # First, group entities by their label
    group_dict = {}
    for label, name in zip(labels, texts):
        group_dict.setdefault(label, []).append(name)

    debug_records = []
    for i, (orig_name, label, final_name) in enumerate(zip(texts, labels, normalized_titles)):
        # List similar names in the group (excluding the current entity)
        similar_names = [name for j, name in enumerate(texts) if labels[j] == label and j != i]
        similar_str = ", ".join(similar_names) if similar_names else ""
        # Compute the maximum similarity score for the current entity with others in the same group
        similar_scores = [sim_matrix[i, j] for j in range(len(texts)) if labels[j] == label and j != i]
        similarity = max(similar_scores) if similar_scores else 1.0
        debug_records.append({
            "Original Entity": orig_name,
            "Similar Entity": similar_str,
            "Similarity Score": similarity,
            "Final Entity": final_name
        })

    debug_df = pd.DataFrame(debug_records)
    debug_df.to_csv("debug_entity_normalization.csv", index=False)

    normalized_df = entities_df.copy()
    normalized_df["title_normalized"] = normalized_titles
    return normalized_df
