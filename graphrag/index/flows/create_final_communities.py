from datetime import datetime, timezone
from uuid import uuid4
import pandas as pd

# --- Original function remains unchanged ---
def create_final_communities(
    base_entity_nodes: pd.DataFrame,
    base_relationship_edges: pd.DataFrame,
    base_communities: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to transform final communities.
    
    Original implementation using a join on 'title'.
    """
    # aggregate entity ids for each community
    entity_ids = base_communities.merge(base_entity_nodes, on="title", how="inner")
    entity_ids = (
        entity_ids.groupby("community").agg(entity_ids=("id", list)).reset_index()
    )

    # aggregate relationships ids for each community
    max_level = int(base_communities["level"].max())
    all_grouped = pd.DataFrame(
        columns=["community", "level", "relationship_ids", "text_unit_ids"]
    )
    for level in range(max_level + 1):
        communities_at_level = base_communities.loc[base_communities["level"] == level]
        sources = base_relationship_edges.merge(
            communities_at_level, left_on="source", right_on="title", how="inner"
        )
        targets = sources.merge(
            communities_at_level, left_on="target", right_on="title", how="inner"
        )
        matched = targets.loc[targets["community_x"] == targets["community_y"]]
        text_units = matched.explode("text_unit_ids")
        grouped = (
            text_units.groupby(["community_x", "level_x", "parent_x"])
            .agg(relationship_ids=("id", list), text_unit_ids=("text_unit_ids", list))
            .reset_index()
        )
        grouped.rename(
            columns={
                "community_x": "community",
                "level_x": "level",
                "parent_x": "parent",
            },
            inplace=True,
        )
        all_grouped = pd.concat([
            all_grouped,
            grouped.loc[
                :, ["community", "level", "parent", "relationship_ids", "text_unit_ids"]
            ],
        ])

    # deduplicate lists in each community.
    all_grouped["relationship_ids"] = all_grouped["relationship_ids"].apply(
        lambda x: sorted(set(x))
    )
    all_grouped["text_unit_ids"] = all_grouped["text_unit_ids"].apply(
        lambda x: sorted(set(x))
    )

    # join and add new fields
    communities = all_grouped.merge(entity_ids, on="community", how="inner")
    communities["id"] = [str(uuid4()) for _ in range(len(communities))]
    communities["human_readable_id"] = communities["community"]
    communities["title"] = "Community " + communities["community"].astype(str)
    communities["parent"] = communities["parent"].astype(int)
    communities["period"] = datetime.now(timezone.utc).date().isoformat()
    communities["size"] = communities.loc[:, "entity_ids"].apply(len)

    return communities.loc[
        :,
        [
            "id",
            "human_readable_id",
            "community",
            "parent",
            "level",
            "title",
            "entity_ids",
            "relationship_ids",
            "text_unit_ids",
            "period",
            "size",
        ],
    ]

# --- New Alternative Implementation ---
def create_final_communities_alternative(
    base_entity_nodes: pd.DataFrame,
    base_relationship_edges: pd.DataFrame,
    base_communities: pd.DataFrame,
) -> pd.DataFrame:
    """Transform final communities in a 'less invasive' way by preserving the original clustering.
    
    Instead of joining on 'title', we use the stable 'community' field so that nodes (e.g. Quote nodes
    attached to an APP node) remain in the intended group.
    """
    #base_communities.to_csv("base_communities.csv", index=False)
    #base_entity_nodes.to_csv("base_entity_nodes.csv", index=False)

    if "community" not in base_entity_nodes.columns:
        # Enrich base_entity_nodes with the community field from base_communities using the title key.
        base_entity_nodes = base_entity_nodes.merge(
            base_communities[["title", "community"]].drop_duplicates(),
            on="title",
            how="left"
        )
    entity_ids = base_communities.merge(base_entity_nodes, on="community", how="inner")

    entity_ids = (
        entity_ids.groupby("community").agg(entity_ids=("id", list)).reset_index()
    )

    max_level = int(base_communities["level"].max())
    all_grouped = pd.DataFrame(
        columns=["community", "level", "relationship_ids", "text_unit_ids"]
    )

    # --- Enrich relationship edges with community info ---
    # base_relationship_edges currently has 'source' and 'target' as the node identifiers (titles).
    # Use the mapping from base_entity_nodes to get the community value.
    mapping = base_entity_nodes[["title", "community"]].drop_duplicates()
    rel_edges = base_relationship_edges.copy()
    rel_edges = rel_edges.merge(mapping, left_on="source", right_on="title", how="left")
    rel_edges.rename(columns={"community": "source_community"}, inplace=True)
    rel_edges.drop(columns=["title"], inplace=True)
    rel_edges = rel_edges.merge(mapping, left_on="target", right_on="title", how="left")
    rel_edges.rename(columns={"community": "target_community"}, inplace=True)
    rel_edges.drop(columns=["title"], inplace=True)

    # Now use these enriched columns for the join.
    # In the for-loop below, we match the relationship edge if both source and target belong
    # to the same community in the base_communities.
    for level in range(max_level + 1):
        communities_at_level = base_communities.loc[base_communities["level"] == level]
        # Convert keys to strings so the merge can proceed without type issues.
        communities_at_level_mod = communities_at_level.copy()
        #communities_at_level_mod["community"] = communities_at_level_mod["community"].astype(int)
        rel_edges_mod = rel_edges.copy()
        rel_edges_mod = rel_edges_mod.dropna(subset=["source_community", "target_community"])
        #rel_edges_mod["source_community"] = rel_edges_mod["source_community"].astype(int)
        #rel_edges_mod["target_community"] = rel_edges_mod["target_community"].astype(int)
        
        # Merge based on the enriched community columns.
        sources = rel_edges_mod.merge(
            communities_at_level_mod, left_on="source_community", right_on="community", how="inner"
        )
        targets = sources.merge(
            communities_at_level_mod, left_on="target_community", right_on="community", how="inner"
        )
        matched = targets.loc[targets["community_x"] == targets["community_y"]]
        text_units = matched.explode("text_unit_ids")
        grouped = (
            text_units.groupby(["community_x", "level_x", "parent_x"])
            .agg(relationship_ids=("id", list), text_unit_ids=("text_unit_ids", list))
            .reset_index()
        )
        grouped.rename(
            columns={
                "community_x": "community",
                "level_x": "level",
                "parent_x": "parent",
            },
            inplace=True,
        )
        all_grouped = pd.concat([
            all_grouped,
            grouped.loc[
                :, ["community", "level", "parent", "relationship_ids", "text_unit_ids"]
            ],
        ])

    # deduplicate the lists
    all_grouped["relationship_ids"] = all_grouped["relationship_ids"].apply(
        lambda x: sorted(set(x))
    )
    all_grouped["text_unit_ids"] = all_grouped["text_unit_ids"].apply(
        lambda x: sorted(set(x))
    )

    # join and add new fields
    communities = all_grouped.merge(entity_ids, on="community", how="inner")
    communities["id"] = [str(uuid4()) for _ in range(len(communities))]
    communities["human_readable_id"] = communities["community"]
    communities["title"] = "Community " + communities["community"].astype(str)
    communities["parent"] = communities["parent"].astype(int)
    communities["period"] = datetime.now(timezone.utc).date().isoformat()
    communities["size"] = communities.loc[:, "entity_ids"].apply(len)

    return communities.loc[
        :,
        [
            "id",
            "human_readable_id",
            "community",
            "parent",
            "level",
            "title",
            "entity_ids",
            "relationship_ids",
            "text_unit_ids",
            "period",
            "size",
        ],
    ]

# --- Wrapper to Choose Implementation ---
def create_final_communities_with_strategy(
    base_entity_nodes: pd.DataFrame,
    base_relationship_edges: pd.DataFrame,
    base_communities: pd.DataFrame,
    strategy: str = "default",  # use "default" or "less_invasive"
) -> pd.DataFrame:
    """Wrapper to select between the default and alternative implementations.
    
    The cluster_graph.strategy setting (or a related config value) can be passed here.
    """
    communities_default = create_final_communities(
        base_entity_nodes, base_relationship_edges, base_communities
    )
    communities_alt = create_final_communities_alternative(
        base_entity_nodes, base_relationship_edges, base_communities
    )
    base_communities.to_csv("debug_base_communities.csv", index=False)
    communities_default.to_csv("debug_final_communities_default.csv", index=False)
    communities_alt.to_csv("debug_final_communities_alt.csv", index=False)
    if strategy == "default":
        return communities_default
    else:
        return communities_alt


