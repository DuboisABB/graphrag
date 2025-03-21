# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing entity_extract methods."""

import logging
from typing import Any

import pandas as pd

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.bootstrap import bootstrap
from graphrag.index.operations.extract_entities.typing import (
    Document,
    EntityExtractStrategy,
    ExtractEntityStrategyType,
)
from graphrag.index.run.derive_from_rows import derive_from_rows
from graphrag.index.operations.entity_normalization import normalize_entities
from graphrag.config.embeddings import get_embedding_settings
from graphrag.config.models.graph_rag_config import GraphRagConfig

log = logging.getLogger(__name__)


DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


async def extract_entities(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    text_column: str,
    id_column: str,
    strategy: dict[str, Any] | None,
    async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types=DEFAULT_ENTITY_TYPES,
    num_threads: int = 4,
    config: GraphRagConfig = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract entities from a piece of text.

    ## Usage
    ```yaml
    args:
        column: the_document_text_column_to_extract_entities_from
        id_column: the_column_with_the_unique_id_for_each_row
        to: the_column_to_output_the_entities_to
        strategy: <strategy_config>, see strategies section below
        summarize_descriptions: true | false /* Optional: This will summarize the descriptions of the entities and relationships, default: true */
        entity_types:
            - list
            - of
            - entity
            - types
            - to
            - extract
    ```

    ## Strategies
    The entity extract verb uses a strategy to extract entities from a document. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### graph_intelligence
    This strategy uses the [graph_intelligence] library to extract entities from a document. In particular it uses a LLM to extract entities from a piece of text. The strategy config is as follows:

    ```yml
    strategy:
        type: graph_intelligence
        extraction_prompt: !include ./entity_extraction_prompt.txt # Optional, the prompt to use for extraction
        completion_delimiter: "<|COMPLETE|>" # Optional, the delimiter to use for the LLM to mark completion
        tuple_delimiter: "<|>" # Optional, the delimiter to use for the LLM to mark a tuple
        record_delimiter: "##" # Optional, the delimiter to use for the LLM to mark a record

        encoding_name: cl100k_base # Optional, The encoding to use for the LLM with gleanings

        llm: # The configuration for the LLM
            type: openai # the type of llm to use, available options are: openai, azure, openai_chat, azure_openai_chat.  The last two being chat based LLMs.
            api_key: !ENV ${GRAPHRAG_OPENAI_API_KEY} # The api key to use for openai
            model: !ENV ${GRAPHRAG_OPENAI_MODEL:gpt-4-turbo-preview} # The model to use for openai
            max_tokens: !ENV ${GRAPHRAG_MAX_TOKENS:6000} # The max tokens to use for openai
            organization: !ENV ${GRAPHRAG_OPENAI_ORGANIZATION} # The organization to use for openai

            # if using azure flavor
            api_base: !ENV ${GRAPHRAG_OPENAI_API_BASE} # The api base to use for azure
            api_version: !ENV ${GRAPHRAG_OPENAI_API_VERSION} # The api version to use for azure
            proxy: !ENV ${GRAPHRAG_OPENAI_PROXY} # The proxy to use for azure

    ```

    ### nltk
    This strategy uses the [nltk] library to extract entities from a document. In particular it uses a nltk to extract entities from a piece of text. The strategy config is as follows:
    ```yml
    strategy:
        type: nltk
    ```
    """
    log.debug("entity_extract strategy=%s", strategy)
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    strategy = strategy or {}
    strategy_exec = _load_strategy(
        strategy.get("type", ExtractEntityStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    num_started = 0

    async def run_strategy(row):
        nonlocal num_started
        text = row[text_column]
        id = row[id_column]
        result = await strategy_exec(
            [Document(text=text, id=id)],
            entity_types,
            callbacks,
            cache,
            strategy_config,
        )
        num_started += 1
        return [result.entities, result.relationships, result.graph]

    results = await derive_from_rows(
        text_units,
        run_strategy,
        callbacks,
        async_type=async_mode,
        num_threads=num_threads,
    )

    entity_dfs = []
    relationship_dfs = []
    for result in results:
        if result:
            entity_dfs.append(pd.DataFrame(result[0]))
            relationship_dfs.append(pd.DataFrame(result[1]))

    entities = _merge_entities(entity_dfs)
    relationships = _merge_relationships(relationship_dfs)
    relationships.to_csv("debug_base_relationships1.csv", index=False)

    normalize_entities_flag = config.entity_extraction.normalize_entities
    normalize_threshold = config.entity_extraction.normalize_threshold

    if normalize_entities_flag:
        # Save a copy of the original title for mapping.
        original_titles = entities["title"].copy()
        missing_titles = original_titles[original_titles.isna()]
        if not missing_titles.empty:
            log.error("Missing title entries detected in entities before normalization: %s", missing_titles)
            print('break')
        entities.to_csv("debug_normalize_entities_before.csv", index=False)
        # Integrate entity normalization to combine entities with similar names.
        entities = await normalize_entities(entities)

        # Debug: log any rows that did not get normalized (NaN in 'title_normalized')
        nan_rows = entities[entities["title_normalized"].isna()]
        if not nan_rows.empty:
            log.error("Normalization failed for the following entities: %s", nan_rows)
            print('break')

        # Build mapping from the original title to the normalized title.
        mapping = dict(zip(original_titles, entities["title_normalized"]))
        mapping_df = pd.DataFrame(list(mapping.items()), columns=["original_title", "normalized_title"])
        mapping_df.to_csv("debug_normalize_entities_mapping.csv", index=False)

        # Reassign normalized title to the title column.
        entities["title"] = entities["title_normalized"]
        # Update relationships to use normalized titles.
        relationships["source"] = relationships["source"].apply(lambda x: mapping.get(x, x))
        relationships["target"] = relationships["target"].apply(lambda x: mapping.get(x, x))

        relationships = relationships.drop_duplicates(subset=["source", "target"])
        # reindex
        relationships = relationships.reset_index(drop=True)


    entities, relationships = _filter_invalid_entities(entities, relationships)

    return (entities, relationships)


def _filter_invalid_entities(entities: pd.DataFrame, relationships: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out entities with empty types and their relationships.
    
    Args:
        entities: DataFrame containing the entities
        relationships: DataFrame containing the relationships
    
    Returns:
        tuple: (filtered_entities, filtered_relationships)
    """
    original_entity_count = len(entities)
    original_relationship_count = len(relationships)
    
    # Filter out entities with empty type
    entities_with_empty_type = entities[entities["type"].isna() | (entities["type"] == "")]
    
    if not entities_with_empty_type.empty:
        # Get the titles of invalid entities
        invalid_entity_titles = set(entities_with_empty_type["title"])
        log.warning(f"Removing {len(invalid_entity_titles)} entities with empty type")
        
        # Remove the invalid entities
        filtered_entities = entities[~entities["title"].isin(invalid_entity_titles)]
        
        # Remove relationships that have invalid entities as source or target
        filtered_relationships = relationships[
            ~relationships["source"].isin(invalid_entity_titles) & 
            ~relationships["target"].isin(invalid_entity_titles)
        ]
        
        # Log the changes
        log.info(f"Removed {original_entity_count - len(filtered_entities)} entities with empty type")
        log.info(f"Removed {original_relationship_count - len(filtered_relationships)} relationships associated with invalid entities")
        
        # Write filtered entities to debug file for inspection
        filtered_entities.to_csv("debug_filtered_entities.csv", index=False)
        filtered_relationships.to_csv("debug_filtered_relationships.csv", index=False)
        
        return filtered_entities, filtered_relationships
    
    return entities, relationships


def _load_strategy(strategy_type: ExtractEntityStrategyType) -> EntityExtractStrategy:
    """Load strategy method definition."""
    match strategy_type:
        case ExtractEntityStrategyType.graph_intelligence:
            from graphrag.index.operations.extract_entities.graph_intelligence_strategy import (
                run_graph_intelligence,
            )

            return run_graph_intelligence

        case ExtractEntityStrategyType.nltk:
            bootstrap()
            # dynamically import nltk strategy to avoid dependency if not used
            from graphrag.index.operations.extract_entities.nltk_strategy import (
                run as run_nltk,
            )

            return run_nltk
        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)


def _merge_entities(entity_dfs) -> pd.DataFrame:
    all_entities = pd.concat(entity_dfs, ignore_index=True)
    return (
        all_entities.groupby(["title", "type"], sort=False)
        .agg(description=("description", list), text_unit_ids=("source_id", list))
        .reset_index()
    )


def _merge_relationships(relationship_dfs) -> pd.DataFrame:
    all_relationships = pd.concat(relationship_dfs, ignore_index=False)
    return (
        all_relationships.groupby(["source", "target"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            weight=("weight", "sum"),
        )
        .reset_index()
    )
