import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity

from graphrag.config.embeddings import get_embedding_settings
from graphrag.config.models.graph_rag_config import GraphRagConfig
from sentence_transformers import SentenceTransformer
from graphrag.callbacks.workflow_callbacks import NoopWorkflowCallbacks

# We need to call embed_text from our embeddings module.
from graphrag.index.operations.embed_text import embed_text

log = logging.getLogger(__name__)


async def normalize_entities(
    entities_df: pd.DataFrame,
    config: GraphRagConfig = None,
    callbacks=None,  # Instance of WorkflowCallbacks (optional)
    cache=None,      # Instance of PipelineCache (optional)
    threshold: float = 0.85
) -> pd.DataFrame:
    """
    Normalize entity names by computing embeddings on a subset of the entities (those with type CHEMICAL or APPLICATION_NAME).
    
    For entities of type CHEMICAL, only the chemical name (the substring before the first dash) is used when deduping,
    and afterwards the remainder of the original title is reattached. For APPLICATION_NAME entities, the entire title is used.

    Parameters
    ----------
    entities_df : pd.DataFrame
        DataFrame with extracted entities. It must have a column 'title' and a column 'type'.
    text_embed_config : dict
        Embedding configuration.
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
    if 'type' not in entities_df.columns:
        raise ValueError("The DataFrame must contain a 'type' column.")

    # Save the extracted entities cache so that debug_entity_normalization can load it.    
    extracted_cache_dir = "./mw/cache/entity_extraction/"
    os.makedirs(extracted_cache_dir, exist_ok=True)
    extracted_cache_path = os.path.join(extracted_cache_dir, "extracted_entities.parquet")
    try:
        entities_df.to_parquet(extracted_cache_path, index=False)
        log.info("Saved extracted entities cache to %s", extracted_cache_path)
    except Exception as e:
        log.exception("Error saving extracted entities cache: %s", e)        

    if callbacks is None:
        callbacks = NoopWorkflowCallbacks()

    # We work only on a subset of rows for which type is CHEMICAL or APPLICATION_NAME.
    mask = entities_df['type'].isin(["CHEMICAL", "APPLICATION_NAME"])
    subset_df = entities_df[mask].copy()

    # For each row, determine the text to normalize.
    # For CHEMICAL type, extract the chemical name (portion before the first " - ").
    # For APPLICATION_NAME, use the full title.
    def get_norm_target(row):
        if row['type'] == "CHEMICAL":
            parts = row['title'].split(" - ", 1)
            return parts[0].strip() if parts else row['title'].strip()
        else:
            return row['title'].strip()

    subset_df['norm_target'] = subset_df.apply(get_norm_target, axis=1)
    texts = subset_df['norm_target'].tolist()

    # Retrieve embedding strategy.
    text_embed_config = get_embedding_settings(config)
    strategy = text_embed_config.get("strategy", {})

    # Prepare the temporary DataFrame required by embed_text.
    tmp_df = pd.DataFrame({
        "id": subset_df.index.tolist(),
        "norm_target": texts
    })

    # Compute embeddings.
    # If a specialized Hugging Face model is defined, use that.
    normalize_model = config.entity_extraction.normalize_model
    if normalize_model == "default_embedding_model":
        
        result_embeddings = await embed_text(
            input=tmp_df,
            callbacks=callbacks,
            cache=cache,
            embed_column="norm_target",
            strategy=strategy,
            embedding_name="entity.title",
            id_column="id",
            title_column="norm_target"
        )
        embeddings = np.array(result_embeddings)
    else:        
        model = SentenceTransformer(normalize_model)
        embeddings = model.encode(texts)

    # Compute cosine similarity matrix among embeddings.    
    sim_matrix = cosine_similarity(embeddings)
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

    # For each group (by label), determine a canonical normalized target.
    label_to_canonical = {}
    for label, text in zip(labels, texts):
        if label not in label_to_canonical:
            label_to_canonical[label] = text

    norm_normalized = [label_to_canonical[label] for label in labels]

    # Create the primary debug dataframe.
    debug_records = []
    for i, (orig_name, label, final_norm) in enumerate(zip(texts, labels, norm_normalized)):
        # Collect similar names from the same group (excluding self) and remove duplicates.
        similar_names = [texts[j] for j in range(n) if labels[j] == label and j != i]
        similar_names = list(dict.fromkeys(similar_names))
        similar_str = ", ".join(similar_names) if similar_names else ""
        similar_scores = [sim_matrix[i, j] for j in range(n) if labels[j] == label and j != i]
        similarity = max(similar_scores) if similar_scores else 1.0
        debug_records.append({
            "Original Entity": orig_name,
            "Similar Entity": similar_str,
            "Similarity Score": similarity,
            "Final Entity": final_norm
        })
    debug_df = pd.DataFrame(debug_records)
    debug_df.to_csv("debug_entity_normalization.csv", index=False)

    # Create a secondary debug dataframe with all entity pairs with similarity >= 0.9.
    pair_debug_records = []
    similarity_threshold_for_debug = 0.9
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if score >= similarity_threshold_for_debug:
                pair_debug_records.append({
                    "Entity 1": texts[i],
                    "Entity 2": texts[j],
                    "Similarity Score": score,
                })
    pair_debug_df = pd.DataFrame(pair_debug_records)
    pair_debug_df.to_csv("debug_entity_similarity_pairs.csv", index=False)

    # For CHEMICAL type entities, reintroduce the remainder of the title.
    def combine_normalized(row, normalized_value):
        if row['type'] == "CHEMICAL":
            parts = row['title'].split(" - ", 1)
            if len(parts) == 2:
                # Append the remainder of the original title.
                return normalized_value + " - " + parts[1].strip()
            else:
                return normalized_value
        else:
            return normalized_value

    # Build a normalized map from the subset indices.
    normalized_map = {}
    for idx, norm_val in zip(subset_df.index, norm_normalized):
        normalized_map[idx] = combine_normalized(subset_df.loc[idx], norm_val)

    # Create the final normalized dataframe and add the 'title_normalized' column.
    normalized_df = entities_df.copy()
    normalized_df["title_normalized"] = normalized_df["title"]  # default to original title.
    for idx, new_title in normalized_map.items():
        normalized_df.at[idx, "title_normalized"] = new_title

    # Save the normalized entities to a parquet cache file.
    cache_dir = "./mw/cache/entity_normalization/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "normalized_entities.parquet")
    try:
        normalized_df.to_parquet(cache_path, index=False)
        log.info("Saved normalized entities cache to %s", cache_path)
    except Exception as e:
        log.exception("Error saving normalized entities cache: %s", e)

    return normalized_df
