from pathlib import Path

from .checks import (
    CANDIDATES,
    LEDGER,
    cost_used,
    decimals_problem,
    gap_problem,
    locked_problem,
    read_json,
    result,
    signature_id_problem,
)
from .rule import parse_decimal


def verify_input(value):
    problem = signature_id_problem(value.dataset, value.signature_id) or decimals_problem(
        threshold=value.threshold, max_jaccard=value.max_jaccard
    )
    if problem:
        return result(False, problem)
    if value.start_rank < 1 or value.count < 1:
        return result(False, "start_rank and count must each be at least 1")
    if value.min_papers < 1 or value.cost_budget < 1:
        return result(False, "min_papers and cost_budget must each be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    ledger = read_json(Path(workspace) / LEDGER)
    if ledger is None or not ledger["checks"]:
        return result(False, "check_pairs left no record of its call in the ledger")
    problem = locked_problem(ledger, input_value)
    if problem:
        return result(False, problem)
    if all(call["signature_id"] != input_value.signature_id for call in ledger["signatures"]):
        return result(False, f"signatures {input_value.signature_id} were never built")
    *earlier, call = ledger["checks"]
    if (call["signature_id"], call["start_rank"], call["count"]) != (
        input_value.signature_id,
        input_value.start_rank,
        input_value.count,
    ):
        return result(False, "the ledger's last check is not this call's signatures and rank range")
    saved = read_json(Path(workspace) / CANDIDATES / f"{input_value.signature_id}.json")
    if saved is None:
        return result(False, f"signatures {input_value.signature_id}'s ranked pairs are not saved in the workspace")
    first = input_value.start_rank
    span = [(a, b, cost) for _, a, b, cost in saved["pairs"][first - 1 : first - 1 + input_value.count]]
    if len(span) != input_value.count:
        return result(False, "the rank range goes past the ranked pairs")
    seen = {(a, b) for past in earlier for a, b in past["checked"]}
    new = [(a, b, cost) for a, b, cost in span if (a, b) not in seen]
    charged = sum(cost for _, _, cost in new)
    if [tuple(pair) for pair in call["checked"]] != [(a, b) for a, b, _ in new]:
        return result(False, "the ledger does not check exactly the range's pairs no earlier call checked")
    if call["skipped"] != len(span) - len(new) or call["charged"] != charged:
        return result(False, f"the range's {len(new)} new pairs cost {charged}: the ledger disagrees")
    if (output_value.checked, output_value.skipped, output_value.cost_charged) != (
        len(new), len(span) - len(new), charged
    ):
        return result(False, f"the range's {len(new)} new pairs cost {charged}: checked, skipped or cost_charged disagree")
    gaps = [{"a": g.a, "b": g.b, "intersection": g.intersection, "union": g.union} for g in output_value.gaps]
    if gaps != call["gaps"]:
        return result(False, "the gaps differ from the ledger's")
    order = {(a, b): rank for rank, (a, b, _) in enumerate(new)}
    ranks = [order.get((g["a"], g["b"])) for g in gaps]
    if None in ranks or ranks != sorted(set(ranks)):
        return result(False, "the gaps are not newly checked pairs of the range, in rank order")
    max_jaccard = parse_decimal(input_value.max_jaccard)
    for gap, (a, b, cost) in zip(gaps, [new[rank] for rank in ranks]):
        problem = gap_problem(gap, max_jaccard)
        if problem:
            return result(False, problem)
        if gap["intersection"] + gap["union"] != cost:
            return result(False, f"pair ({a!r}, {b!r}) has |A| + |B| = {cost}, not intersection + union")
    used = cost_used(ledger)
    total = len(seen) + len(new)
    if (output_value.total_checked, output_value.cost_used, output_value.cost_remaining) != (
        total, used, input_value.cost_budget - used
    ):
        return result(False, f"{total} pairs are checked and {used} spent: totals disagree")
    if used > input_value.cost_budget:
        return result(False, f"{used} is spent, over the cost budget of {input_value.cost_budget}")
    return result(True, "the range is checked within budget", checked=len(new), gaps=len(gaps))
