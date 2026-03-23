# ADR-004: Blocking Strategy for Scalable Entity Dedup

**Status**: accepted
**Date**: 2026-03-23

## Context

Cross-document dedup requires comparing new entity names against all existing graph entities. Naive O(N*M) comparison is too slow for large graphs (M = 100K+). Need a sub-quadratic approach.

## Decision

Use **multi-key blocking** — a standard technique from record linkage / entity resolution — to partition entities into small buckets before detailed comparison.

## Blocking Keys

For each entity name, generate multiple blocking keys. Two names are candidates for comparison only if they share at least one blocking key.

```python
def _build_blocking_keys(name: str) -> set[str]:
    """Generate blocking keys for entity name comparison."""
    keys = set()
    words = name.split()
    
    # Key 1: Each individual word (catches "DVORKIN" in both 
    # "ALEXANDER DVORKIN" and "ALEKSANDR DVORKIN")
    for w in words:
        if len(w) >= 3:  # skip short words like "OF", "THE"
            keys.add(f"w:{w}")
    
    # Key 2: First 3 chars of each word (catches transliteration: 
    # "ALE" from ALEXANDER/ALEKSANDR)
    for w in words:
        if len(w) >= 3:
            keys.add(f"p3:{w[:3]}")
    
    # Key 3: Consonant skeleton of full name
    skeleton = _consonant_skeleton(name)
    if len(skeleton) >= 3:
        keys.add(f"cs:{skeleton}")
    
    # Key 4: Sorted words (catches reordering: 
    # "DVORKIN ALEXANDER" vs "ALEXANDER DVORKIN")
    if len(words) >= 2:
        keys.add(f"sw:{' '.join(sorted(words))}")
    
    return keys
```

## Expected Performance

| Graph Size | New Entities | Naive Comparisons | With Blocking | Speedup |
|-----------|-------------|-------------------|---------------|---------|
| 1K        | 50          | 50K               | ~2K           | 25x     |
| 10K       | 50          | 500K              | ~5K           | 100x    |
| 100K      | 50          | 5M                | ~15K          | 333x    |

Blocking key generation: O(M) one-time build of inverted index.
Candidate generation: O(N * avg_block_size).
Detailed comparison: O(candidates) with existing SequenceMatcher + heuristics.

## Data Structure

```python
# Build inverted index: blocking_key → set of entity names
blocking_index: dict[str, set[str]] = defaultdict(set)
for name in existing_entity_names:
    for key in _build_blocking_keys(name):
        blocking_index[key].add(name)

# For each new entity, find candidates
for new_name in new_entity_names:
    candidates = set()
    for key in _build_blocking_keys(new_name):
        candidates.update(blocking_index.get(key, set()))
    # Now compare new_name only against candidates (not all M)
```

## Consequences

**Pros**: Reduces comparison count by 25-333x. Memory overhead is O(M * avg_keys_per_name) ≈ O(4M) — negligible.

**Cons**: Blocking can miss pairs that share no blocking key (false negatives). Mitigated by using 4 independent key types — a pair must differ in ALL words, ALL prefixes, consonant skeleton, AND sorted order to be missed. This is extremely unlikely for true duplicates.

## Files to Modify

1. **`lightrag/operate.py`** — Add `_build_blocking_keys()`, `_consonant_skeleton()`. Use in `resolve_cross_document_entities()`.
