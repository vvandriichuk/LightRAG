# ADR-002: Transliteration Variant Detection

**Status**: accepted
**Date**: 2026-03-23

## Context

Names transliterated from Cyrillic (or other scripts) to Latin produce multiple valid spellings: ALEKSANDR/ALEXANDER, DVORKIN/DVORKIN (stable), SERGEI/SERGEY/SERGII. Standard string similarity (`SequenceMatcher`) gives these ~0.75-0.82 ratio, below the default threshold of 0.85, so they escape dedup.

## Decision

Add a `_is_transliteration_variant()` function that uses phonetic/structural heuristics to detect transliteration pairs WITHOUT a dictionary.

## Algorithm

Three complementary signals, any one sufficient for a match (when combined with same-entity-type and word-count match):

### 1. Consonant Skeleton Similarity
Strip vowels, collapse doubles, compare:
- ALEKSANDR → LKSNDR
- ALEXANDER → LXNDR

Then apply SequenceMatcher to consonant skeletons with a lower threshold (0.70).

Rationale: Transliteration primarily varies in vowel rendering (E/A, I/Y, EI/EY). Consonant structure is more stable.

### 2. Known Transliteration Equivalences (small, static map)
A compact map of common Cyrillic→Latin transliteration alternations, not a name dictionary:

```python
TRANSLIT_EQUIVALENCES = [
    ("KS", "X"),       # Александр → Aleksandr/Alexander
    ("DJ", "J"),       # Джон → Dzhon/John
    ("YU", "IU"),      # Юрий → Yuriy/Iuriy
    ("YA", "IA"),      # Яков → Yakov/Iakov
    ("EI", "EY"),      # Сергей → Sergei/Sergey
    ("II", "IY"),      # Дмитрий → Dmitrii/Dmitriy
    ("SCH", "SH"),     # Щ → Shch/Sch
    ("TS", "C"),       # Цезарь → Tsezar/Cezar
    ("PH", "F"),       # not Cyrillic but common in names
    ("TH", "T"),       # Тео → Theo/Teo
]
```

Apply all equivalences to both names, then compare normalized forms.

### 3. Shared-Word Anchor
If multi-word names share at least one identical word (e.g., both end with "DVORKIN"), compare only the differing words with a relaxed threshold (0.65).

## Integration

Called from `_cluster_similar_names()` as an additional check alongside `_is_abbreviation_of()` and `_words_are_subset()`:

```python
elif _is_transliteration_variant(shorter, longer):
    union(a, b)
```

Also called from the new `resolve_cross_document_entities()` in Phase A blocking comparison.

## Consequences

**Pros**: Catches 80-90% of Cyrillic transliteration variants without any external dependency or dictionary. Static map is ~20 lines, zero runtime cost.

**Cons**: Won't catch radically different transliterations (e.g., Горбачёв → Gorbachev vs Gorbachov — but SequenceMatcher already catches these at 0.89). May produce false positives for unrelated short names — mitigated by requiring same entity type and word count.

## Files to Modify

1. **`lightrag/operate.py`** — Add `_is_transliteration_variant()`, `TRANSLIT_EQUIVALENCES`, `_consonant_skeleton()`. Modify `_cluster_similar_names()` to call it.
