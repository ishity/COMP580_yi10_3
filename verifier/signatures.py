from .checks import (
    candidates_problem,
    charges_problem,
    cost_used,
    decimals_problem,
    histogram_of,
    key_of,
    locked_problem,
    read_candidates,
    read_ledger,
    result,
)
from .rule import HASH_ROWS


def verify_input(value):
    problem = decimals_problem(threshold=value.threshold, max_jaccard=value.max_jaccard)
    if problem:
        return result(False, problem)
    if not 1 <= value.rows <= HASH_ROWS:
        return result(False, f"rows must be from 1 to {HASH_ROWS}")
    if value.min_papers < 1 or value.cost_budget < 1:
        return result(False, "min_papers and cost_budget must each be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    expected = f"{input_value.dataset.value}-r{input_value.rows}"
    if output_value.signature_id != expected:
        return result(False, f"signature_id must be {expected}, not {output_value.signature_id}")
    key = key_of(input_value)
    if key is None:
        return result(False, "threshold and max_jaccard must be decimal strings")
    ledger = read_ledger(workspace, key)
    if ledger is None or not ledger["signatures"]:
        return result(False, "build_signatures left no record of its call in the ledger")
    problem = locked_problem(ledger, input_value) or charges_problem(ledger)
    if problem:
        return result(False, problem)
    call = ledger["signatures"][-1]
    if (call["signature_id"], call["rows"]) != (expected, input_value.rows):
        return result(False, "the ledger's last signatures are not this call's")
    used = cost_used(ledger)
    if (output_value.cost_charged, output_value.cost_used, output_value.cost_remaining) != (
        call["charged"],
        used,
        input_value.cost_budget - used,
    ):
        return result(False, f"{used} is spent: cost_charged, cost_used or cost_remaining disagree")
    if used > input_value.cost_budget:
        return result(False, f"{used} is spent, over the cost budget of {input_value.cost_budget}")
    saved = read_candidates(workspace, key, expected)
    if saved is None:
        return result(False, f"signatures {expected}'s ranked pairs are not saved in the workspace")
    problem = candidates_problem(saved, expected, input_value)
    if problem:
        return result(False, problem)
    if output_value.related_pairs != len(saved["pairs"]):
        return result(False, f"output ranks {output_value.related_pairs} pairs, the signatures saved {len(saved['pairs'])}")
    given = [
        (row.min_matches, row.max_matches, row.min_estimate, row.max_estimate, row.pairs, row.check_cost,
         row.first_rank, row.last_rank)
        for row in output_value.histogram
    ]
    if given != histogram_of(saved):
        return result(False, "the histogram does not match the saved pairs")
    return result(True, "the related pairs are ranked and charged", related=output_value.related_pairs)
