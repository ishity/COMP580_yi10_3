from .checks import (
    cost_used,
    decimals_problem,
    gap_problem,
    key_of,
    locked_problem,
    read_candidates,
    read_ledger,
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
    key = key_of(input_value)
    if key is None:
        return result(False, "threshold and max_jaccard must be decimal strings")
    ledger = read_ledger(workspace, key)
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
    saved = read_candidates(workspace, key, input_value.signature_id)
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
    # The newly found gaps the ledger records: newly checked pairs of the range, in rank order, gaps by their integers.
    order = {(a, b): rank for rank, (a, b, _) in enumerate(new)}
    ranks = [order.get((g["a"], g["b"])) for g in call["gaps"]]
    if None in ranks or ranks != sorted(set(ranks)):
        return result(False, "the ledger's gaps are not newly checked pairs of the range, in rank order")
    max_jaccard = parse_decimal(input_value.max_jaccard)
    for gap, rank in zip(call["gaps"], ranks):
        problem = gap_problem(gap, max_jaccard)
        if problem:
            return result(False, problem)
        if gap["intersection"] + gap["union"] != new[rank][2]:
            return result(False, f"pair ({gap['a']!r}, {gap['b']!r}) has |A| + |B| = {new[rank][2]}, not intersection + union")
    # The output: every gap of the range in rank order, those an earlier call found with its integers, marked.
    fresh = {(g["a"], g["b"]): g for g in call["gaps"]}
    scored = {(g["a"], g["b"]): g for past in earlier for g in past["gaps"]}
    expected = []
    for a, b, _ in span:
        if (a, b) in seen:
            if (a, b) in scored:
                expected.append(scored[(a, b)] | {"previously_scored": True})
        elif (a, b) in fresh:
            expected.append(fresh[(a, b)] | {"previously_scored": False})
    given = [
        {"a": g.a, "b": g.b, "intersection": g.intersection, "union": g.union, "previously_scored": g.previously_scored}
        for g in output_value.gaps
    ]
    if given != expected:
        return result(False, "the gaps differ from the range's gaps the ledger records, new and previously scored")
    used = cost_used(ledger)
    total = len(seen) + len(new)
    if (output_value.total_checked, output_value.cost_used, output_value.cost_remaining) != (
        total, used, input_value.cost_budget - used
    ):
        return result(False, f"{total} pairs are checked and {used} spent: totals disagree")
    if used > input_value.cost_budget:
        return result(False, f"{used} is spent, over the cost budget of {input_value.cost_budget}")
    return result(True, "the range is checked within budget", checked=len(new), gaps=len(expected))
