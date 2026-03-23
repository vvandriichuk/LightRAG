"""
One-time script to deduplicate existing entities in the graph.
Run AFTER deploying cross-doc dedup code (Steps 1-3).

Usage:
    python scripts/dedup_existing_graph.py

Requires:
    - .env with PG connection settings in project root
    - LightRAG installed (pip install -e .)

Algorithm:
    1. Fetch all entity names from graph via get_all_labels()
    2. Cluster using the same dedup functions from operate.py
    3. For each cluster, pick canonical name (most connected = highest degree)
    4. For each non-canonical name:
       a. Merge description/source_ids into canonical node
       b. Re-point all edges from duplicate to canonical
       c. Delete duplicate node from graph
    5. Update VDB entries accordingly
"""

from __future__ import annotations

import asyncio
import difflib
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag import LightRAG, QueryParam
from lightrag.constants import (
    DEFAULT_ENTITY_DEDUP_THRESHOLD,
    GRAPH_FIELD_SEP,
)
from lightrag.operate import (
    _cluster_similar_names,
    _is_abbreviation_of,
    _is_transliteration_variant,
    _words_are_subset,
    _is_garbage_entity,
)
from lightrag.utils import logger, compute_mdhash_id

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)


async def get_rag_instance() -> LightRAG:
    """Create a LightRAG instance from environment variables."""
    from lightrag.llm.openai import openai_embed

    rag = LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./data/rag_storage"),
        kv_storage=os.getenv("KV_STORAGE", "PGKVStorage"),
        vector_storage=os.getenv("VECTOR_STORAGE", "PGVectorStorage"),
        graph_storage=os.getenv("GRAPH_STORAGE", "PGGraphStorage"),
        doc_status_storage=os.getenv("DOC_STATUS_STORAGE", "PGDocStatusStorage"),
    )
    await rag.initialize_storages()
    return rag


async def run_cleanup(dry_run: bool = True):
    """Run entity dedup cleanup on existing graph."""
    rag = await get_rag_instance()
    graph = rag.chunk_entity_relation_graph

    logger.info("Fetching all entity names from graph...")
    all_labels = await graph.get_all_labels()
    logger.info(f"Found {len(all_labels)} entities in graph")

    if not all_labels:
        logger.info("No entities found, nothing to clean up.")
        await rag.finalize_storages()
        return

    # Step 1: Remove garbage entities
    garbage_names = [name for name in all_labels if _is_garbage_entity(name)]
    if garbage_names:
        logger.info(f"Found {len(garbage_names)} garbage entities to remove")
        for name in garbage_names:
            logger.info(f"  Garbage: '{name}'")
            if not dry_run:
                try:
                    await graph.delete_node(name)
                    eid = compute_mdhash_id(name, prefix="ent-")
                    await rag.entities_vdb.delete_entity(eid)
                    logger.info(f"  Deleted: '{name}'")
                except Exception as e:
                    logger.warning(f"  Failed to delete '{name}': {e}")

    # Step 2: Cluster remaining entities
    clean_labels = [n for n in all_labels if n not in set(garbage_names)]
    threshold = float(os.getenv("ENTITY_DEDUP_THRESHOLD", str(DEFAULT_ENTITY_DEDUP_THRESHOLD)))

    logger.info(f"Clustering {len(clean_labels)} entities (threshold={threshold})...")
    clusters = _cluster_similar_names(clean_labels, threshold)

    if not clusters:
        logger.info("No duplicate clusters found.")
        await rag.finalize_storages()
        return

    logger.info(f"Found {len(clusters)} clusters of duplicates")

    # Step 3: For each cluster, merge into canonical
    total_merged = 0
    for cluster in clusters:
        # Pick canonical: highest degree (most connected)
        degrees = {}
        for name in cluster:
            try:
                degrees[name] = await graph.node_degree(name)
            except Exception:
                degrees[name] = 0

        # Sort by degree desc, then name length desc
        scored = sorted(cluster, key=lambda n: (degrees.get(n, 0), len(n)), reverse=True)
        canonical = scored[0]
        duplicates = scored[1:]

        logger.info(f"\nCluster: {cluster}")
        logger.info(f"  Canonical: '{canonical}' (degree={degrees.get(canonical, 0)})")

        if dry_run:
            for dup in duplicates:
                logger.info(f"  Would merge: '{dup}' (degree={degrees.get(dup, 0)}) -> '{canonical}'")
            total_merged += len(duplicates)
            continue

        # Merge each duplicate into canonical
        canonical_node = await graph.get_node(canonical)
        if not canonical_node:
            logger.warning(f"  Canonical node '{canonical}' not found in graph, skipping")
            continue

        for dup in duplicates:
            try:
                dup_node = await graph.get_node(dup)
                if not dup_node:
                    logger.info(f"  Duplicate '{dup}' already removed, skipping")
                    continue

                # Merge descriptions
                canon_desc = canonical_node.get("description", "")
                dup_desc = dup_node.get("description", "")
                if dup_desc and dup_desc not in canon_desc:
                    merged_desc = f"{canon_desc}{GRAPH_FIELD_SEP}{dup_desc}" if canon_desc else dup_desc
                    canonical_node["description"] = merged_desc

                # Merge source_ids
                canon_src = set(canonical_node.get("source_id", "").split(GRAPH_FIELD_SEP))
                dup_src = set(dup_node.get("source_id", "").split(GRAPH_FIELD_SEP))
                canonical_node["source_id"] = GRAPH_FIELD_SEP.join(
                    s for s in canon_src | dup_src if s
                )

                # Update canonical node
                await graph.upsert_node(canonical, canonical_node)

                # Re-point edges
                edges = await graph.get_node_edges(dup)
                if edges:
                    for src, tgt in edges:
                        try:
                            edge_data = await graph.get_edge(src, tgt)
                            if edge_data:
                                new_src = canonical if src == dup else src
                                new_tgt = canonical if tgt == dup else tgt
                                if new_src != new_tgt:
                                    await graph.upsert_edge(new_src, new_tgt, edge_data)
                        except Exception as e:
                            logger.warning(f"  Failed to re-point edge ({src}, {tgt}): {e}")

                # Delete duplicate node
                await graph.delete_node(dup)

                # Update VDB
                dup_eid = compute_mdhash_id(dup, prefix="ent-")
                try:
                    await rag.entities_vdb.delete_entity(dup_eid)
                except Exception:
                    pass

                logger.info(f"  Merged: '{dup}' -> '{canonical}'")
                total_merged += 1

            except Exception as e:
                logger.error(f"  Error merging '{dup}' -> '{canonical}': {e}")

    mode = "DRY RUN" if dry_run else "COMPLETED"
    logger.info(f"\n{'='*50}")
    logger.info(f"{mode}: {total_merged} entities merged across {len(clusters)} clusters")
    if dry_run:
        logger.info("Run with --apply to execute changes")

    await rag.finalize_storages()


def main():
    dry_run = "--apply" not in sys.argv
    if dry_run:
        print("Running in DRY RUN mode. Use --apply to execute changes.")
        print("IMPORTANT: Back up your database before running with --apply!")
    else:
        print("Running in APPLY mode. Changes will be written to the database.")

    asyncio.run(run_cleanup(dry_run=dry_run))


if __name__ == "__main__":
    main()
