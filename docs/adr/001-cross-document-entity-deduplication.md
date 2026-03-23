# ADR-001: Cross-Document Entity Deduplication

**Status**: accepted
**Date**: 2026-03-23

## Context

Current entity deduplication in LightRAG works only within a single document's extraction batch. When Document1 creates "ALEXANDER DVORKIN" and Document2 creates "ALEKSANDR DVORKIN", they never meet in `resolve_entity_duplicates()` because that function receives only `all_nodes` from the current document's chunks. This leads to persistent duplicates that accumulate as the graph grows.

## Decision

Implement a **hybrid two-phase approach**: extend the existing intra-document dedup with a cross-document lookup phase that runs BEFORE merge, using the graph's existing `get_all_labels()` method to fetch current entity names and include them in the clustering.

## Architecture: Two-Phase Cross-Document Dedup

### Phase Overview

```
Document chunks extracted
        │
        ▼
┌─────────────────────────────┐
│ Phase A: Cross-Document     │  ← NEW
│ Match new names against     │
│ existing graph entity names │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Phase B: Intra-Document     │  ← EXISTING (unchanged)
│ Cluster new names among     │
│ themselves                  │
└─────────────────────────────┘
        │
        ▼
   merge_nodes_and_edges()
```

### Phase A: Cross-Document Matching (new code)

**Location**: New function `resolve_cross_document_entities()` in `operate.py`, called from `merge_nodes_and_edges()` BEFORE the existing `resolve_entity_duplicates()`.

**Algorithm**:

1. Get all existing entity names from graph via `knowledge_graph_inst.get_all_labels()`.
2. For each new entity name (from current document extraction), find candidates from existing names using a **two-tier filter**:
   - **Tier 1 — Cheap blocking**: Group names by "blocking keys" to avoid O(N*M) full comparison. Blocking keys:
     - First+last word (e.g., "ALEXANDER DVORKIN" → "ALEXANDER|DVORKIN")
     - Sorted consonant skeleton (e.g., "ALEKSANDR" → "LKSNDR", "ALEXANDER" → "LXNDR") — shared suffix "DR" + first letter
     - First 3 characters + last 3 characters (catches transliteration variants)
   - **Tier 2 — Detailed comparison** (only within same block):
     - Reuse existing `_is_abbreviation_of()`, `_words_are_subset()`
     - `difflib.SequenceMatcher` with existing threshold
     - NEW: `_is_transliteration_variant()` (see ADR-002)
3. Build mapping: `new_name → existing_canonical_name`.
4. For ambiguous matches (multiple existing candidates), use LLM resolution (reuse `_llm_resolve_entity_clusters`).
5. Apply mapping to `all_nodes` and `all_edges` via existing `_apply_name_mapping()`.

**Complexity**: O(N * B) where N = new entities, B = average block size. With blocking, B << M (total existing entities). For 10K existing entities, typical B ≈ 5-20.

### Phase B: Intra-Document Matching (existing, unchanged)

After Phase A remaps new names to existing ones where possible, the remaining new names (those without an existing match) still go through `resolve_entity_duplicates()` to catch duplicates among themselves.

### Integration Point

In `merge_nodes_and_edges()` at line ~2849, BEFORE the existing dedup call:

```python
# ===== Cross-Document Entity Dedup Phase ===== (NEW)
enable_entity_dedup = global_config.get("enable_entity_dedup", DEFAULT_ENABLE_ENTITY_DEDUP)
if enable_entity_dedup and len(all_nodes) > 0:
    all_nodes, all_edges = await resolve_cross_document_entities(
        all_nodes, all_edges, knowledge_graph_inst, global_config, llm_response_cache
    )

# ===== Intra-Document Entity Dedup Phase ===== (EXISTING)
if enable_entity_dedup and len(all_nodes) > 1:
    all_nodes, all_edges = await resolve_entity_duplicates(
        dict(all_nodes), dict(all_edges), global_config, llm_response_cache
    )
```

### Graph Update Protocol (when renaming to existing entity)

When a new entity name maps to an existing graph entity, NO special graph update is needed — `_merge_nodes_then_upsert()` already handles this correctly:
- It calls `knowledge_graph_inst.get_node(entity_name)` for the canonical name
- If the node exists, it merges descriptions, source_ids, etc.
- The VDB entry is upserted with the canonical name's ID

The old (duplicate) name never reaches the graph because `_apply_name_mapping()` rewrites `all_nodes` keys and `all_edges` src/tgt BEFORE merge.

## Why This Approach

1. **Minimal invasiveness**: Only adds one new function call before the existing dedup. Does not change `_merge_nodes_then_upsert`, VDB upsert logic, or any storage interface.
2. **Uses existing abstractions**: `get_all_labels()` is already implemented in all storage backends (PG, Neo4j, NetworkX, etc.).
3. **Blocking avoids O(N*M)**: With 10K existing entities and 50 new entities per document, naive comparison = 500K. With blocking ≈ 1-5K comparisons.
4. **LLM fallback for ambiguity**: Reuses the existing `_llm_resolve_entity_clusters()` for cases where string matching is uncertain.

## Alternatives Considered

- **Variant A (get_all_nodes)** — Rejected: returns full node properties (description, source_ids), too heavy. `get_all_labels()` returns just names, which is all we need.
- **Variant B (new get_all_entity_names method)** — Rejected: `get_all_labels()` already does exactly this. Adding a new abstract method breaks all storage backends.
- **Variant C (VDB vector similarity for names)** — Rejected for primary matching: entity name embeddings capture semantic meaning, not string similarity. "ALEKSANDR" and "ALEXANDER" may have very different embeddings. However, VDB can be a supplementary signal (see Future Work).
- **Variant D (post-processing after all documents)** — Rejected: requires graph node renaming (delete old + create new + update all edges), which is complex and risky. Catching duplicates BEFORE merge is much simpler.

## Consequences

**Pros**:
- Eliminates the primary source of cross-document duplicates
- Zero changes to storage interfaces (BaseGraphStorage, BaseVectorStorage, etc.)
- Incremental: works per-document, no batch post-processing needed
- Compatible with all storage backends

**Cons**:
- `get_all_labels()` cost grows with graph size. For 100K entities on PG, this is ~50ms (single column SELECT with index). Acceptable.
- Blocking may miss some matches (false negatives). Mitigated by multiple blocking keys and LLM fallback.
- First-document-wins: the canonical name is whichever form was indexed first. This is acceptable since `_pick_canonical_name` prefers longer/more frequent names, and the graph already has the "winning" form.

## Files to Modify

1. **`lightrag/operate.py`** — Add `resolve_cross_document_entities()`, `_build_blocking_keys()`, `_is_transliteration_variant()`. Modify `merge_nodes_and_edges()` to call new function.
2. **`lightrag/constants.py`** — Add `DEFAULT_ENABLE_CROSS_DOC_DEDUP = True`.
3. **`lightrag/prompt.py`** — No changes (reuses existing `entity_resolution_prompt`).

All files are volume-mountable in Docker.
