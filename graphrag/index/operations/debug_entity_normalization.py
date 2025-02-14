#poetry run python -m graphrag.index.operations.debug_entity_normalization  
import asyncio
import sys
import os
import pandas as pd
import yaml

from graphrag.index.operations.entity_normalization import normalize_entities
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.config.read_dotenv import read_dotenv

# Monkey-patch get_norm_target so it returns a single value per row.
def patched_get_norm_target(row):
    result = get_norm_target(row)
    # if result is a DataFrame or Series with more than one value, take the first element
    if isinstance(result, pd.DataFrame):
        # Assume the first column is the desired output.
        return result.iloc[0, 0]
    elif isinstance(result, pd.Series) and result.shape[0] > 1:
        return result.iloc[0]
    return result

#import graphrag.index.operations.entity_normalization as en
#en.get_norm_target = patched_get_norm_target

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


async def unit_test_entity_normalization():
    """
    Run a unit test mode with explicit entity name pairs.
    Test data includes four columns:
      - name1
      - name2
      - same (1 indicates they should be merged, 0 otherwise)
      - replacement (the preferred normalized name, or 'null' if not applicable)

    For each test, we create a DataFrame with the two names (in a column 'title'),
    run normalize_entities using the current logic, and then compare the resulting
    normalized names.
    """
    # Define hard-coded unit test cases.
    test_cases = [
        {"name1": "CO", "name2": "carbon monoxide", "expected_same": True, "expected_replacement": "CO"},
        {"name1": "CO2", "name2": "carbon dioxide", "expected_same": True, "expected_replacement": "CO2"},
        {"name1": "Water", "name2": "H2O", "expected_same": True, "expected_replacement": "Water"},
        {"name1": "H2S", "name2": "hydrogen sulfide", "expected_same": True, "expected_replacement": "H2S"},
        {"name1": "VCM", "name2": "vinyl chloride", "expected_same": True, "expected_replacement": "VCM"},
        {"name1": "n-Butane", "name2": "C4H10", "expected_same": True, "expected_replacement": "Butane"},
        {"name1": "Butane", "name2": "C4H10", "expected_same": True, "expected_replacement": "Butane"},
        {"name1": "EDC", "name2": "1,2-Dichloroethane", "expected_same": True, "expected_replacement": "EDC"},
        {"name1": "Benzene", "name2": "C6H6", "expected_same": True, "expected_replacement": "Benzene"},
        {"name1": "1,2-DICHLOROETHANE", "name2": "1,1-DICHLOROETHANE", "expected_same": False, "expected_replacement": "null"},
    ]

    # Load the configuration to pass along to normalize_entities.
    cfg_path = os.path.expanduser("~/graphrag/mw/settings.yaml")
    with open(cfg_path, "r", encoding="utf-8") as config_file:
        raw_config = config_file.read()
        expanded_config = os.path.expandvars(raw_config)
        config_dict = yaml.safe_load(expanded_config)
    config = create_graphrag_config(values=config_dict)

    print("Running unit tests for entity normalization...\n")
    for idx, tc in enumerate(test_cases, start=1):
        # Create a DataFrame with two rows representing the two entity names.
        df = pd.DataFrame({
            "title": [tc["name1"], tc["name2"]],
            "type": ["APPLICATION_NAME", "APPLICATION_NAME"]
        })
        # Run the normalization logic contained in normalize_entities.
        normalized_df = await normalize_entities(
            entities_df=df,
            config=config,
            threshold=0.98
        )

        # Determine if the normalization merged the rows.
        if tc["expected_same"]:
            if len(normalized_df) != 1:
                print(
                    f"[FAIL] Test {idx}: Expected names to be merged for '{tc['name1']}' and '{tc['name2']}', but got {len(normalized_df)} rows."
                )
                # Print detailed info about the normalized DataFrame.
                print("Normalized DataFrame contents:")
                print(normalized_df.to_string(index=False))
            else:
                result_name = normalized_df.iloc[0]["title"]
                if result_name != tc["expected_replacement"]:
                    print(
                        f"[FAIL] Test {idx}: Merged name '{result_name}' does not match expected replacement '{tc['expected_replacement']}' for '{tc['name1']}' and '{tc['name2']}'."
                    )
                    print("Normalized DataFrame contents:")
                    print(normalized_df.to_string(index=False))
                else:
                    print(
                        f"[PASS] Test {idx}: '{tc['name1']}' and '{tc['name2']}' merged into '{result_name}' as expected."
                    )
        else:
            if len(normalized_df) == 1:
                result_name = normalized_df.iloc[0]["title"]
                print(
                    f"[FAIL] Test {idx}: Expected names to remain separate for '{tc['name1']}' and '{tc['name2']}', but they merged into '{result_name}'."
                )
                print("Normalized DataFrame contents:")
                print(normalized_df.to_string(index=False))
            elif len(normalized_df) == 2:
                print(
                    f"[PASS] Test {idx}: '{tc['name1']}' and '{tc['name2']}' remained separate as expected."
                )
            else:
                print(
                    f"[FAIL] Test {idx}: Unexpected number of rows ({len(normalized_df)}) for '{tc['name1']}' and '{tc['name2']}'."
                )
                print("Normalized DataFrame contents:")
                print(normalized_df.to_string(index=False))
    print("\nUnit tests completed.")


async def main():
    # Check for unit test mode.
    if len(sys.argv) > 1 and sys.argv[1] == "--unit-test":
        await unit_test_entity_normalization()
        return

    # Define cache file paths for entity extraction results.
    csv_cache_path = "./mw/cache/entity_extraction/entities.csv"
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
    with open(cfg_dir, "r", encoding="utf-8") as config_file:
        raw_config = config_file.read()
        expanded_config = os.path.expandvars(raw_config)
        config_dict = yaml.safe_load(expanded_config)
    config = create_graphrag_config(values=config_dict)

    # Run entity normalization.
    normalized_df = await normalize_entities(
        entities_df=entities_df,
        config=config,
        threshold=0.98,
        save_parquet_cache=False
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
