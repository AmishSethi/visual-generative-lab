"""Representative query subsampling for the rebuttal evaluators.

`evaluate_table2.maybe_truncate_queries` keeps a prefix, `values[:N]`.  For size,
rotation and count the query lists are short enough that this is harmless.  For
position it is not: the grid is built as nested loops over x then y, so a prefix
is the left-hand columns of the canvas rather than a sample of it.  Capping
position that way would report accuracy over a spatially biased slice and
silently mis-state the number.

`evenly_spaced` takes N items spread across the whole list instead, which keeps
the full spatial extent and stays deterministic.
"""
import numpy as np


def evenly_spaced(values, n):
    """N items spread across `values`, preserving order and endpoints."""
    if n is None or len(values) <= n:
        return list(values)
    idx = np.linspace(0, len(values) - 1, n).round().astype(int)
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(values[i])
    return out


def truncate_queries(queries, max_conditions_per_split):
    """Drop-in replacement for evaluate_table2.maybe_truncate_queries."""
    if max_conditions_per_split is None:
        return queries
    return {split: evenly_spaced(values, max_conditions_per_split)
            for split, values in queries.items()}


def install(evaluate_table2_module):
    """Patch the paper evaluator to subsample evenly rather than by prefix."""
    evaluate_table2_module.maybe_truncate_queries = truncate_queries
