# Entity Deduplication: Implementation Plan

## Summary

This document provides a concrete implementation plan for cross-document entity deduplication in LightRAG, based on the architecture decisions in `docs/adr/001-004`.

---

## Step 1: Garbage Entity Post-Filter (ADR-003)

**Priority**: HIGH — prevents new garbage from entering the graph.
**Effort**: ~1 hour.
**File**: `lightrag/operate.py`

### What to add

After line ~408 in `_handle_single_entity_extraction()`, after all normalization steps:

```python
import re

# Module-level constants (top of file, near other constants)
GARBAGE_ENTITY_PREPOSITIONS = frozenset({"OF", "FOR", "BY", "FROM", "ABOUT", "AGAINST", "WITHIN"})
GARBAGE_ENTITY_MAX_WORDS = 7

GARBAGE_ENTITY_PATTERNS = [
    re.compile(r"'S\b"),           # residual possessive
    re.compile(r"\bAGENT OF\b"),
    re.compile(r"\bMEMBER OF\b"),
    re.compile(r"\bGOAL OF\b"),
    re.compile(r"\bPART OF\b"),
    re.compile(r"\bFOLLOWER OF\b"),
    re.compile(r"\bLEADER OF\b"),
    re.compile(r"\b\d{4}:\s"),     # year-prefixed document titles
    re.compile(r"\bSECT\.\s*\d"),  # section references: "SECT. 2009"
]


def _is_garbage_entity(name: str) -> bool:
    """Return True if entity name matches known garbage patterns."""
    words = name.split()
    if len(words) > GARBAGE_ENTITY_MAX_WORDS:
        return True

    if len(words) >= 3:
        prep_count = sum(1 for w in words if w in GARBAGE_ENTITY_PREPOSITIONS)
        if prep_count >= 1 and prep_count / len(words) > 0.25:
            return True

    for pattern in GARBAGE_ENTITY_PATTERNS:
        if pattern.search(name):
            return True

    return False
```

### Integration point

```python
# In _handle_single_entity_extraction(), after line ~408
if _is_garbage_entity(entity_name):
    logger.info(f"Filtered garbage entity: '{entity_name}'")
    return None
```

---

## Step 2: Transliteration Detection (ADR-002)

**Priority**: HIGH — required for cross-doc dedup to catch Cyrillic transliterations.
**Effort**: ~1.5 hours.
**File**: `lightrag/operate.py`

### What to add

```python
# Module-level constants
TRANSLIT_EQUIVALENCES: list[tuple[str, str]] = [
    ("KS", "X"),
    ("CK", "K"),
    ("DJ", "J"),
    ("DZH", "J"),
    ("YU", "IU"),
    ("YA", "IA"),
    ("YE", "IE"),
    ("EI", "EY"),
    ("II", "IY"),
    ("IJ", "IY"),
    ("SCH", "SH"),
    ("SHCH", "SH"),
    ("TS", "C"),
    ("TZ", "C"),
    ("PH", "F"),
    ("TH", "T"),
    ("OV", "OFF"),
    ("EV", "EFF"),
    ("W", "V"),
]

VOWELS = frozenset("AEIOUY")


def _consonant_skeleton(name: str) -> str:
    """Strip vowels and collapse repeated consonants."""
    result = []
    prev = ""
    for ch in name:
        if ch in VOWELS or ch == " ":
            prev = ""
            continue
        if ch != prev:
            result.append(ch)
        prev = ch
    return "".join(result)


def _normalize_translit(name: str) -> str:
    """Apply transliteration equivalences to normalize a name."""
    result = name
    for a, b in TRANSLIT_EQUIVALENCES:
        result = result.replace(a, b)
        result = result.replace(b, a)  # bidirectional normalization
    # Use the lexicographically smaller form for consistency
    result2 = name
    for a, b in TRANSLIT_EQUIVALENCES:
        result2 = result2.replace(b, a)
    # Pick the canonical normalization: apply all left-to-right
    canonical = name
    for a, b in TRANSLIT_EQUIVALENCES:
        # Normalize to the shorter form
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        canonical = canonical.replace(longer, shorter)
    return canonical


def _is_transliteration_variant(name_a: str, name_b: str) -> bool:
    """Check if two names are likely transliteration variants."""
    words_a = name_a.split()
    words_b = name_b.split()

    # Must have same word count
    if len(words_a) != len(words_b):
        return False

    # Must share at least one identical word (anchor)
    shared = set(words_a) & set(words_b)
    if len(words_a) > 1 and not shared:
        return False

    # Compare consonant skeletons
    skel_a = _consonant_skeleton(name_a)
    skel_b = _consonant_skeleton(name_b)
    skel_sim = difflib.SequenceMatcher(None, skel_a, skel_b).ratio()
    if skel_sim >= 0.70:
        return True

    # Compare after transliteration normalization
    norm_a = _normalize_translit(name_a)
    norm_b = _normalize_translit(name_b)
    if norm_a == norm_b:
        return True
    norm_sim = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
    if norm_sim >= 0.85:
        return True

    return False
```

### Integration in `_cluster_similar_names()`

Add after the `_words_are_subset` check (around line 2582):

```python
elif _is_transliteration_variant(a, b):
    union(a, b)
```

---

## Step 3: Cross-Document Entity Dedup (ADR-001 + ADR-004)

**Priority**: CRITICAL — the core fix.
**Effort**: ~3 hours.
**File**: `lightrag/operate.py`, `lightrag/constants.py`

### constants.py addition

```python
DEFAULT_ENABLE_CROSS_DOC_DEDUP = True
```

### operate.py — new functions

```python
def _build_blocking_keys(name: str) -> set[str]:
    """Generate blocking keys for entity name comparison."""
    keys = set()
    words = name.split()

    for w in words:
        if len(w) >= 3:
            keys.add(f"w:{w}")
            keys.add(f"p3:{w[:3]}")

    skeleton = _consonant_skeleton(name)
    if len(skeleton) >= 3:
        keys.add(f"cs:{skeleton}")

    if len(words) >= 2:
        keys.add(f"sw:{' '.join(sorted(words))}")

    return keys


async def resolve_cross_document_entities(
    all_nodes: dict[str, list],
    all_edges: dict[tuple, list],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[dict[str, list], dict[tuple, list]]:
    """Match new entity names against existing graph entities for cross-document dedup.

    Uses blocking to avoid O(N*M) comparison, then applies string similarity,
    abbreviation detection, transliteration detection, and LLM resolution.
    """
    new_names = list(all_nodes.keys())
    if not new_names:
        return all_nodes, all_edges

    # Fetch all existing entity names from graph
    existing_names = await knowledge_graph_inst.get_all_labels()
    if not existing_names:
        return all_nodes, all_edges

    # Filter out names that already exist in graph (these will merge naturally)
    existing_set = set(existing_names)
    truly_new = [n for n in new_names if n not in existing_set]
    if not truly_new:
        return all_nodes, all_edges

    # Build blocking index from existing names
    blocking_index: dict[str, set[str]] = defaultdict(set)
    for name in existing_names:
        for key in _build_blocking_keys(name):
            blocking_index[key].add(name)

    threshold = global_config.get("entity_dedup_threshold", DEFAULT_ENTITY_DEDUP_THRESHOLD)

    # Find candidates for each new name
    simple_mapping: dict[str, str] = {}
    ambiguous_pairs: list[list[str]] = []

    for new_name in truly_new:
        # Find candidates via blocking
        candidates: set[str] = set()
        for key in _build_blocking_keys(new_name):
            candidates.update(blocking_index.get(key, set()))

        if not candidates:
            continue

        # Score each candidate
        best_match: str | None = None
        best_score: float = 0.0
        is_deterministic = False

        for existing_name in candidates:
            shorter, longer = (new_name, existing_name) if len(new_name) <= len(existing_name) else (existing_name, new_name)

            # Check deterministic matchers first
            if _is_abbreviation_of(shorter, longer):
                best_match = existing_name
                is_deterministic = True
                break

            if _words_are_subset(shorter, longer):
                best_match = existing_name
                is_deterministic = True
                break

            if _is_transliteration_variant(new_name, existing_name):
                best_match = existing_name
                is_deterministic = True
                break

            # String similarity
            sim = difflib.SequenceMatcher(None, new_name, existing_name).ratio()
            if sim >= threshold and sim > best_score:
                best_score = sim
                best_match = existing_name

        if best_match:
            if is_deterministic or best_score >= threshold:
                simple_mapping[new_name] = best_match
                logger.info(
                    f"Cross-doc dedup: '{new_name}' -> '{best_match}' "
                    f"({'deterministic' if is_deterministic else f'sim={best_score:.2f}'})"
                )
            else:
                ambiguous_pairs.append([new_name, best_match])

    # LLM resolution for ambiguous pairs
    llm_mapping = {}
    if ambiguous_pairs:
        logger.info(f"Cross-doc dedup: sending {len(ambiguous_pairs)} ambiguous pairs to LLM")
        llm_mapping = await _llm_resolve_entity_clusters(
            ambiguous_pairs, global_config, llm_response_cache
        )
        for variant, canonical in llm_mapping.items():
            logger.info(f"Cross-doc dedup (LLM): '{variant}' -> '{canonical}'")

    full_mapping = {**simple_mapping, **llm_mapping}
    if not full_mapping:
        return all_nodes, all_edges

    logger.info(f"Cross-doc dedup: remapping {len(full_mapping)} entity names to existing graph entities")
    return _apply_name_mapping(all_nodes, all_edges, full_mapping)
```

### Integration in `merge_nodes_and_edges()`

At line ~2849, BEFORE existing dedup:

```python
# ===== Cross-Document Entity Dedup Phase =====
enable_cross_doc_dedup = global_config.get("enable_cross_doc_dedup", DEFAULT_ENABLE_CROSS_DOC_DEDUP)
enable_entity_dedup = global_config.get("enable_entity_dedup", DEFAULT_ENABLE_ENTITY_DEDUP)
if enable_cross_doc_dedup and enable_entity_dedup and len(all_nodes) > 0:
    all_nodes, all_edges = await resolve_cross_document_entities(
        dict(all_nodes), dict(all_edges),
        knowledge_graph_inst, global_config, llm_response_cache
    )

# ===== Intra-Document Entity Dedup Phase ===== (existing, unchanged)
if enable_entity_dedup and len(all_nodes) > 1:
    all_nodes, all_edges = await resolve_entity_duplicates(
        dict(all_nodes), dict(all_edges), global_config, llm_response_cache
    )
```

---

## Step 4: Retroactive Cleanup (one-time script)

**Priority**: MEDIUM — cleans up existing duplicates from before the fix.
**Effort**: ~2 hours.
**Approach**: Standalone Python script (NOT modifying LightRAG core).

### Script: `scripts/dedup_existing_graph.py`

```python
"""
One-time script to deduplicate existing entities in the graph.
Run AFTER deploying the cross-doc dedup code.

Algorithm:
1. Fetch all entity names from graph via get_all_labels()
2. Cluster using _cluster_similar_names() + _is_transliteration_variant()
3. For each cluster, pick canonical name (most connected = highest degree)
4. For each non-canonical name in cluster:
   a. Fetch the non-canonical node data
   b. Merge description/source_ids into canonical node
   c. Update all edges referencing the non-canonical name
   d. Delete non-canonical node from graph + VDB + KV
   e. Upsert updated canonical node to graph + VDB + KV
"""
```

### Graph update order (for safety):

1. **Read** both nodes (canonical + duplicate)
2. **Merge** descriptions, source_ids, file_paths into canonical
3. **Update** canonical node in graph (`upsert_node`)
4. **Re-point edges**: for each edge touching the duplicate:
   - Get edge data
   - Delete old edge (`remove_edges`)
   - Create new edge with canonical name (`upsert_edge`)
5. **Delete** duplicate node from graph (`delete_node`)
6. **Update VDB**: delete duplicate entity VDB entry (`delete_entity`), upsert canonical (`upsert`)
7. **Update KV**: merge entity_chunks entries, delete duplicate key
8. **Update relationship VDB**: delete old relationship entries, upsert with new src/tgt names

This order ensures no data loss — canonical node is updated BEFORE duplicate is deleted. If script crashes mid-way, re-running is safe (idempotent because we check if duplicate still exists).

---

## Execution Order

| Step | What | Risk | Rollback |
|------|------|------|----------|
| 1 | Garbage filter | Low — only filters new extractions | Remove the `_is_garbage_entity` call |
| 2 | Transliteration detection | Low — only adds matches, never removes | Remove `_is_transliteration_variant` call from `_cluster_similar_names` |
| 3 | Cross-doc dedup | Medium — remaps entity names before merge | Set `enable_cross_doc_dedup=False` in config |
| 4 | Retroactive cleanup | Medium — modifies graph | Take PG backup before running |

Steps 1-2 can be deployed independently and immediately.
Step 3 depends on Step 2 (uses transliteration detection).
Step 4 should run after Step 3 is deployed and verified.

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_entity_dedup` | `True` | Master switch for all dedup |
| `enable_cross_doc_dedup` | `True` | Cross-document dedup (new) |
| `entity_dedup_threshold` | `0.85` | String similarity threshold |

---

## Testing Strategy

### Unit tests for new functions:
- `_is_garbage_entity()`: known garbage names, edge cases (BANK OF AMERICA must pass)
- `_is_transliteration_variant()`: ALEKSANDR/ALEXANDER, SERGEI/SERGEY, non-matches
- `_consonant_skeleton()`: various inputs
- `_build_blocking_keys()`: verify expected keys generated
- `resolve_cross_document_entities()`: mock `get_all_labels()`, verify mapping

### Integration test:
- Insert doc1 with "ALEXANDER DVORKIN", insert doc2 with "ALEKSANDR DVORKIN"
- Verify graph has single node "ALEXANDER DVORKIN" (first-indexed wins as canonical)
- Verify descriptions from both documents are merged

---

## Docker Volume Mounts

Only these files need to be mounted:
1. `lightrag/operate.py` — all dedup logic
2. `lightrag/constants.py` — new default constants
3. `scripts/dedup_existing_graph.py` — one-time cleanup (optional)
