"""describe_corpus: how big a bundled corpus is, how many concepts are frequent and pairs related, and what costs."""

from __future__ import annotations

from pathlib import Path

from contract import DescribeInput, DescribeOutput
from plugin.bundle import decimal, frequent, load, positive, postings, related_pairs
from plugin.rule import HASH_ROWS, ROW_RATE


def describe_corpus(given: DescribeInput, workspace: Path) -> DescribeOutput:
    positive(given.min_papers, "min_papers")
    threshold = decimal(given.threshold, "threshold")
    corpus = load(given.dataset)
    related = related_pairs(given.dataset, given.min_papers, threshold)
    return DescribeOutput(
        dataset=given.dataset,
        papers=corpus.paper_count,
        concepts=len(corpus.names),
        frequent_concepts=len(frequent(corpus, given.min_papers)),
        frequent_postings=postings(corpus, given.min_papers),
        related_pairs=len(related),
        related_check_cost=sum(pair.check_cost for pair in related),
        hash_rows=HASH_ROWS,
        row_rate=ROW_RATE,
    )
