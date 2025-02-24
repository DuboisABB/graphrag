import numpy as np
import pandas as pd
import os
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity

from graphrag.config.embeddings import get_embedding_settings
from graphrag.config.models.graph_rag_config import GraphRagConfig
from sentence_transformers import SentenceTransformer
#from graphrag.callbacks.workflow_callbacks import NoopWorkflowCallbacks
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

# We need to call embed_text from our embeddings module.
from graphrag.index.operations.embed_text import embed_text

from collections import defaultdict

log = logging.getLogger(__name__)


async def normalize_entities(
    entities_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize CHEMICAL entity names. We group identical chemicals that have a similar nominal concentration.
    First extract the chemical name using this regex: "CHEM - (.+?)(?= - \d+\.\d+ NOM)"

    Group all identical chemicals together.
    Then extract the nominal concentration using this regex: "CHEM -.+ - ([\d\.]+) NOM"
    The concentration will be in the 0 to 100 range. We want to create 3 groups, with a roughly equal number of chemicals in each group.
    So, we needs to calculate these group boundaries dynamically for each chemical. But we have rules for the boundaries:
    - Group 1: lower bound: 0, upper bound: <= 10
    - Group 2: lower bound: >= 5, upper bound: <= 50
    - Group 3: lower bound: >= 30, upper bound: <= 100
    One a boundaries for a given chemical is found, we use it to create 3 groups and assign each chemical node to either of those 3 groups. For example, we could have these 3 groups:
    - CHEM - NITROGEN - MATRIX_OR_INT - (0 to 8.0) NOM
    - CHEM - NITROGEN - MATRIX_OR_INT - (8.0 to 45) NOM
    - CHEM - NITROGEN - MATRIX_OR_INT - (45 to 100) NOM
    Then we need to fix the relationships to match the new chemical nodes names

    Parameters
    ----------
    entities_df : pd.DataFrame
        DataFrame with extracted entities. It must have a column 'title' and a column 'type'.
    
    Returns
    -------
    pd.DataFrame
        A copy of entities_df with an additional column 'title_normalized'.
    """
    # Verify required columns exist.
    if 'title' not in entities_df.columns:
        raise ValueError("The DataFrame must contain a 'title' column.")
    if 'type' not in entities_df.columns:
        raise ValueError("The DataFrame must contain a 'type' column.")

    normalized_df = entities_df.copy()
    # Default normalized title is the original title.
    normalized_df["title_normalized"] = normalized_df["title"]

    # Process only CHEMICAL entities.
    chem_mask = normalized_df['type'] == "CHEMICAL"
    chem_df = normalized_df.loc[chem_mask].copy()

    # Dictionaries to capture extraction results.
    # For each CHEMICAL row, we record:
    # - base: extracted chemical name according to regex.
    # - conc: nominal concentration as float.
    # - remainder: any additional identifier (if available) between the chemical name and concentration.
    base_names = {}
    concentrations = {}
    remainders = {}

    # Define regex patterns.
    base_pattern = re.compile(r"CHEM - (.+?)(?= - \d+\.\d+ NOM)")
    conc_pattern = re.compile(r"CHEM -.+ - ([\d\.]+) NOM")

    for idx, row in chem_df.iterrows():
        title = row["title"]
        base_match = base_pattern.search(title)
        conc_match = conc_pattern.search(title)
        if base_match and conc_match:
            base = base_match.group(1).strip()
            try:
                conc = float(conc_match.group(1))
            except ValueError:
                conc = None
        else:
            # Fallback: use entire title as base and no concentration.
            base = title
            conc = None

        base_names[idx] = base
        concentrations[idx] = conc

        # Attempt to extract a "remainder" if available.
        parts = title.split(" - ")
        # Expecting format like: CHEM - <base> - <identifier> - (<concentration>) NOM.
        if len(parts) >= 3:
            remainders[idx] = parts[2].strip()
        else:
            remainders[idx] = ""

    # Group CHEMICAL entities by their extracted base name.
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, base in base_names.items():
        if concentrations[idx] is not None:
            groups[base].append((idx, concentrations[idx]))
        else:
            # If concentration extraction failed, put in its own group.
            groups[base].append((idx, None))

    # For each chemical group, compute boundaries if possible and update normalized title.    
    for base, items in groups.items():
        # Extract items with valid concentration values.
        valid_items = [(idx, conc) for idx, conc in items if conc is not None]
        if valid_items:
            # Compute approximate quantiles to split into 3 groups.
            concentrations_list = [conc for _, conc in valid_items]
            sorted_concs = sorted(concentrations_list)
            n_valid = len(sorted_concs)
            q1_idx = n_valid // 3
            q2_idx = (2 * n_valid) // 3

            # Default boundaries from quantiles.
            boundary1 = sorted_concs[q1_idx] if n_valid > 0 else 10
            boundary2 = sorted_concs[q2_idx] if n_valid > 1 else 50

            # Enforce hard rules:
            # Group 1: 0 to <= 10, so boundary1 must not be above 10.
            boundary1 = min(boundary1, 10)
            # Group 2: lower bound >= 5 and upper bound <= 50; we choose boundary2 clamped between 30 and 50.
            boundary2 = max(min(boundary2, 50), 30)
        else:
            # If no valid concentrations, fallback boundaries.
            boundary1, boundary2 = 10, 50

        # Update normalized title for each group member.
        for idx, conc in items:
            if conc is None:
                # If no concentration information, keep the original title.
                continue

            # Determine the group for this concentration.
            if conc < boundary1:
                new_title = f"CHEM - {base} - {remainders.get(idx, '')} - (0 to {boundary1}) NOM"
            elif conc < boundary2:
                new_title = f"CHEM - {base} - {remainders.get(idx, '')} - ({boundary1} to {boundary2}) NOM"
            else:
                new_title = f"CHEM - {base} - {remainders.get(idx, '')} - ({boundary2} to 100) NOM"

            # Update normalized_df.
            normalized_df.at[idx, "title_normalized"] = new_title.strip(" -")
    # In this refactored function, relationships updating should happen outside of normalize_entities.
    return normalized_df

async def normalize_entities_old(
    entities_df: pd.DataFrame,
    config: GraphRagConfig = None,
    callbacks=None,  # Instance of WorkflowCallbacks (optional)
    cache=None,      # Instance of PipelineCache (optional)
    threshold: float = 0.98,
    save_parquet_cache: bool = True,
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

    if save_parquet_cache:
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

    if save_parquet_cache:
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

if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    import pandas as pd

    # Load entity names from the parquet file.
    file_path = Path.home() / "graphrag" / "mw" / "output" / "create_final_entities.parquet"
    try:
        entities_df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        raise

    # Run the normalization function.
    normalized_df = asyncio.run(normalize_entities(entities_df))

    # Save the resulting normalized entities to a debug file.
    debug_entities_file = "debug_normalized_entities.csv"
    normalized_df.to_csv(debug_entities_file, index=False)
    print(f"Normalized entities saved to {debug_entities_file}")

    # Create a summary dataframe of the number of nodes in each title_normalized group.
    summary_df = (
        normalized_df.groupby("title_normalized")
        .size()
        .reset_index(name="node_count")
    )

    # Save the debug summary to file.
    debug_summary_file = "debug_title_normalized_summary.csv"
    summary_df.to_csv(debug_summary_file, index=False)
    print(f"Normalized title summary saved to {debug_summary_file}")
