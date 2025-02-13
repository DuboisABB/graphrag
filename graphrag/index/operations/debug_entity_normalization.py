import asyncio
import sys
import os
import pandas as pd
import yaml

from graphrag.index.operations.entity_normalization import normalize_entities
from graphrag.config.create_graphrag_config import create_graphrag_config

from graphrag.config.read_dotenv import read_dotenv

# Expand ~ to the full path
env_dir = os.path.expanduser("~/graphrag/mw")
read_dotenv(env_dir)

openai_key = os.environ.get("GRAPHRAG_API_KEY")
if openai_key:
    print(f"DEBUG: GRAPHRAG_API_KEY starts with: {openai_key[:5]}")
else:
    print("DEBUG: GRAPHRAG_API_KEY not found")

if os.environ.get("GRAPHRAG_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["GRAPHRAG_API_KEY"]
    print("Set OPENAI_API_KEY from GRAPHRAG_API_KEY")    

async def main():
    # Define cache file paths for entity extraction results.
    # Our primary cache may be produced as CSV (if available)…
    csv_cache_path = "./mw/cache/entity_extraction/entities.csv"
    # …or, if empty or not parseable, fall back to our own parquet cache.
    parquet_cache_path = "./mw/cache/entity_extraction/extracted_entities.parquet"

    entities_df = pd.DataFrame()
    if os.path.exists(csv_cache_path):
        try:
            entities_df = pd.read_csv(csv_cache_path)
        except Exception as e:
            print(f"Error loading CSV cache '{csv_cache_path}': {e}")
    
    if entities_df.empty and os.path.exists(parquet_cache_path):
        try:
            entities_df = pd.read_parquet(parquet_cache_path)
            print(f"Loaded entities from parquet cache '{parquet_cache_path}'")
        except Exception as e:
            print(f"Error loading parquet cache '{parquet_cache_path}': {e}")
    
    if entities_df.empty:
        print("Could not load valid entity extraction cache. Exiting.")
        sys.exit(1)


    # Load config file from "./mw/settings.yaml" and use create_graphrag_config to generate config.
    cfg_dir = os.path.expanduser("~/graphrag/mw/settings.yaml")
    # Read the file as a string and substitute environment variables.
    with open(cfg_dir, "r", encoding="utf-8") as config_file:
        raw_config = config_file.read()
        expanded_config = os.path.expandvars(raw_config)
        config_dict = yaml.safe_load(expanded_config)

    config = create_graphrag_config(values=config_dict)
    #print(config)


    # Run entity normalization. This function will:
    # 1. Generate a group-level debug CSV ("debug_entity_normalization.csv")
    # 2. Generate a pair-level debug CSV for entities above a chosen threshold ("debug_entity_similarity_pairs.csv")
    normalized_df = await normalize_entities(
        entities_df=entities_df,
        config=config,
        threshold=0.98
    )

    # Save the normalized entities in our cache as a parquet file for future use.
    normalized_cache_path = "./mw/cache/entity_normalization/normalized_entities.parquet"
    try:
        normalized_df.to_parquet(normalized_cache_path, index=False)
        print(f"Normalized entities saved to '{normalized_cache_path}'")
    except Exception as e:
        # Fallback to CSV if parquet fails.
        normalized_csv_path = "./mw/cache/entity_extraction/normalized_entities.csv"
        normalized_df.to_csv(normalized_csv_path, index=False)
        print(f"Normalized entities saved to '{normalized_csv_path}'")

    print("Debug CSV files 'debug_entity_normalization.csv' and 'debug_entity_similarity_pairs.csv' have been generated.")

if __name__ == "__main__":
    asyncio.run(main())
