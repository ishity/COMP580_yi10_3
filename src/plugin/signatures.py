"""build_signatures: MinHash signatures of the frequent concepts, and every related pair ranked by estimated Jaccard.

Each frequent concept's signature holds, for each of the first ``rows`` hash rows (a, b), the least
(a * x + b) mod p over its papers x. A related pair's estimate is the share of rows where the two signatures
match: the chance that one row matches is the pair's Jaccard similarity. The related pairs go to
candidates/<ledger_key>/<signature_id>.json, fewest matching rows first (most likely gaps first), ties in (a, b) order, each
with its matches and its exact check cost |A| + |B|, so that they never pass through the model.

Every call is recorded in ledgers/<ledger_key>.json, the ledger named by the task values, and the ranking is saved
under that ledger, so that instances sharing a workspace never share a ledger or a ranking. The first call fixes the dataset, min_papers, threshold, max_jaccard and
cost_budget. Signatures are charged incrementally: the first rows of longer signatures are the shorter ones, so a
build is charged only for rows beyond the longest built so far, and a build at or below it is charged 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from contract import MatchBucket, SignaturesInput, SignaturesOutput
from plugin.bundle import (
    LOCKED,
    candidates_path,
    check_locked,
    cost_used,
    decimal,
    ledger_key,
    load,
    positive,
    postings,
    read_ledger,
    related_pairs,
    signature,
    signature_id,
    write_ledger,
)
from plugin.rule import HASH_ROWS, histogram, signature_charge, signature_cost


def _ledger(given: SignaturesInput, workspace: Path, key: str, total: int) -> dict:
    """The ledger so far, or a new one fixing this call's task values."""
    ledger = read_ledger(workspace, key)
    if ledger is not None:
        check_locked(ledger, given)
        return ledger
    fixed = {field: getattr(given, field) for field in LOCKED} | {"dataset": given.dataset.value}
    return fixed | {"frequent_postings": total, "signatures": [], "checks": []}


def build_signatures(given: SignaturesInput, workspace: Path) -> SignaturesOutput:
    if not 1 <= given.rows <= HASH_ROWS:
        raise ValueError(f"rows must be from 1 to {HASH_ROWS}, the hash rows there are, got {given.rows}")
    positive(given.min_papers, "min_papers")
    positive(given.cost_budget, "cost_budget")
    threshold = decimal(given.threshold, "threshold")
    decimal(given.max_jaccard, "max_jaccard")

    corpus = load(given.dataset)
    total = postings(corpus, given.min_papers)
    key = ledger_key(given)
    ledger = _ledger(given, workspace, key, total)
    built = max((call["rows"] for call in ledger["signatures"]), default=0)
    charged = signature_charge(given.rows, built, total)
    used = cost_used(ledger)
    if used + charged > given.cost_budget:
        left = given.cost_budget - used
        fits = built
        while fits < HASH_ROWS and signature_cost(fits + 1, total) - signature_cost(built, total) <= left:
            fits += 1
        raise ValueError(
            f"{given.rows}-row signatures cost {charged} more (P = {total}, longest built {built} rows), but only"
            f" {left} of the cost budget {given.cost_budget} remain, since {used} is spent: "
            + (f"rows may be at most {fits}" if fits else "no signatures fit")
            + (f"; up to {built} rows cost nothing more" if built else "")
        )

    sig = signature_id(given.dataset, given.rows)
    ranked = []
    for pair in related_pairs(given.dataset, given.min_papers, threshold):
        first, second = signature(given.dataset, pair.i), signature(given.dataset, pair.j)
        matches = sum(1 for k in range(given.rows) if first[k] == second[k])
        ranked.append((matches, pair.a, pair.b, pair.check_cost))
    ranked.sort()

    path = candidates_path(workspace, key, sig)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "signature_id": sig,
        "dataset": given.dataset.value,
        "rows": given.rows,
        "min_papers": given.min_papers,
        "threshold": given.threshold,
        "pairs": [list(row) for row in ranked],
    }
    path.write_text(json.dumps(saved) + "\n")
    ledger["signatures"].append({"signature_id": sig, "rows": given.rows, "charged": charged})
    write_ledger(workspace, key, ledger)

    buckets = histogram([(matches, cost) for matches, _, _, cost in ranked], given.rows)
    return SignaturesOutput(
        signature_id=sig,
        cost_charged=charged,
        cost_used=used + charged,
        cost_remaining=given.cost_budget - used - charged,
        related_pairs=len(ranked),
        histogram=[MatchBucket(*bucket) for bucket in buckets],
    )
