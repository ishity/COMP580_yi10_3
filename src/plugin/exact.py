"""exact_gaps: every related pair of a corpus, measured exactly, and the gaps among them, the task's reference."""

from __future__ import annotations

from pathlib import Path

from contract import ExactOutput, FindInput, MeasuredPair, Pair
from plugin.bundle import decimal, load, positive, postings, related_pairs
from plugin.rule import is_gap


def exact_gaps(given: FindInput, workspace: Path) -> ExactOutput:
    positive(given.min_papers, "min_papers")
    positive(given.cost_budget, "cost_budget")
    threshold = decimal(given.threshold, "threshold")
    max_jaccard = decimal(given.max_jaccard, "max_jaccard")
    decimal(given.min_recall, "min_recall")
    corpus = load(given.dataset)
    related = sorted(related_pairs(given.dataset, given.min_papers, threshold), key=lambda pair: (pair.a, pair.b))
    return ExactOutput(
        dataset=given.dataset,
        min_papers=given.min_papers,
        threshold=given.threshold,
        max_jaccard=given.max_jaccard,
        concepts=list(corpus.names),
        frequent_postings=postings(corpus, given.min_papers),
        related=[MeasuredPair(pair.a, pair.b, pair.intersection, pair.union) for pair in related],
        gaps=[Pair(pair.a, pair.b) for pair in related if is_gap(pair.intersection, pair.union, max_jaccard)],
    )
