"""check_pairs: measure a rank range of the related pairs exactly, and return the gaps, within the cost budget.

Every call is recorded in ledger.json, which build_signatures started, and must pass the task values it fixed.
Checking a pair costs |A| + |B|, the posting reads of merging its two sorted paper lists. A pair any earlier call
checked is skipped, neither checked nor charged again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contract import CheckInput, CheckOutput, MeasuredPair
from plugin.bundle import (
    candidates_path,
    check_locked,
    cost_used,
    decimal,
    load,
    positive,
    read_ledger,
    signature_id_pattern,
    write_ledger,
)
from plugin.rule import is_gap


def check_pairs(given: CheckInput, workspace: Path) -> CheckOutput:
    positive(given.min_papers, "min_papers")
    positive(given.cost_budget, "cost_budget")
    decimal(given.threshold, "threshold")
    max_jaccard = decimal(given.max_jaccard, "max_jaccard")
    if given.start_rank < 1 or given.count < 1:
        raise ValueError(f"start_rank and count must each be at least 1, got {given.start_rank} and {given.count}")
    if not re.fullmatch(signature_id_pattern(given.dataset), given.signature_id):
        raise ValueError(
            f"signature_id {given.signature_id!r} is not a {given.dataset.value} signature:"
            f" pass a signature_id build_signatures returned for it, {given.dataset.value}-r<rows>"
        )
    ledger = read_ledger(workspace)
    if ledger is None:
        raise ValueError("no signatures are built yet: call build_signatures first, then check its ranks")
    check_locked(ledger, given)
    path = candidates_path(workspace, given.signature_id)
    if not path.exists() or all(call["signature_id"] != given.signature_id for call in ledger["signatures"]):
        raise ValueError(f"signatures {given.signature_id} are not built: call build_signatures for them first")
    ranked = json.loads(path.read_text())["pairs"]
    first, last = given.start_rank, given.start_rank + given.count - 1
    if first > len(ranked):
        raise ValueError(
            f"{given.signature_id} ranks {len(ranked)} related pairs: start_rank must be at most {len(ranked)}"
        )
    if last > len(ranked):
        raise ValueError(
            f"ranks {first} to {last} go past {given.signature_id}'s {len(ranked)} related pairs:"
            f" with start_rank {first}, count may be at most {len(ranked) - first + 1}"
        )

    seen = {(a, b) for call in ledger["checks"] for a, b in call["checked"]}
    used = cost_used(ledger)
    remaining = given.cost_budget - used
    span = [(a, b, cost) for _, a, b, cost in ranked[first - 1 : last]]
    new = [(a, b, cost) for a, b, cost in span if (a, b) not in seen]
    charged = sum(cost for _, _, cost in new)
    if charged > remaining:
        fits, spent = 0, 0
        for a, b, cost in span:
            spent += 0 if (a, b) in seen else cost
            if spent > remaining:
                break
            fits += 1
        blocked = next(cost for a, b, cost in span[fits:] if (a, b) not in seen)
        raise ValueError(
            f"ranks {first} to {last} hold {len(new)} pairs not yet checked ({len(span) - len(new)} already checked),"
            f" costing {charged}, but only {remaining} of the cost budget {given.cost_budget} remain: "
            + (f"with start_rank {first}, count may be at most {fits}" if fits else
               f"the pair at rank {first} alone costs {blocked}, so no range from rank {first} fits")
            + ("; the cost budget is used up, so submit what you found" if not remaining else "")
        )

    corpus = load(given.dataset)
    index = {name: i for i, name in enumerate(corpus.names)}
    gaps = []
    for a, b, _ in new:
        held_a, held_b = corpus.papers[index[a]], corpus.papers[index[b]]
        inter = len(set(held_a) & set(held_b))
        union = len(held_a) + len(held_b) - inter
        if is_gap(inter, union, max_jaccard):
            gaps.append(MeasuredPair(a=a, b=b, intersection=inter, union=union))
    ledger["checks"].append(
        {
            "signature_id": given.signature_id,
            "start_rank": first,
            "count": given.count,
            "checked": [[a, b] for a, b, _ in new],
            "skipped": len(span) - len(new),
            "charged": charged,
            "gaps": [{"a": g.a, "b": g.b, "intersection": g.intersection, "union": g.union} for g in gaps],
        }
    )
    write_ledger(workspace, ledger)
    total = len(seen) + len(new)
    return CheckOutput(
        gaps=gaps,
        checked=len(new),
        skipped=len(span) - len(new),
        cost_charged=charged,
        total_checked=total,
        cost_used=used + charged,
        cost_remaining=remaining - charged,
    )
