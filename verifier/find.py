"""The find-concept-gaps task's output check, against the exact_gaps reference and the ledger.

The first failing check decides: malformed pairs, the budget, precision, provenance, then recall.
"""

from fractions import Fraction
from pathlib import Path

from .checks import CANDIDATES, LEDGER, charges_problem, cost_used, locked_problem, read_json, result
from .rule import parse_decimal


def verify_task(arguments, output, reference, workspace):
    """``arguments``: a contract FindInput; ``output``: a FindOutput; ``reference``: exact_gaps' ExactOutput."""
    if (reference.dataset, reference.min_papers) != (arguments.dataset, arguments.min_papers) or (
        parse_decimal(reference.threshold), parse_decimal(reference.max_jaccard)
    ) != (parse_decimal(arguments.threshold), parse_decimal(arguments.max_jaccard)):
        return result(False, "the reference is for another dataset, min_papers, threshold or max_jaccard than the task")
    min_recall = parse_decimal(arguments.min_recall)
    if min_recall is None:
        return result(False, "min_recall must be a decimal string in (0, 1]")

    # 1. Malformed: two distinct known concepts per pair, each pair once in either order.
    concepts = set(reference.concepts)
    submitted = []
    for pair in output.pairs:
        if pair.a == pair.b:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) pairs a concept with itself")
        for concept in (pair.a, pair.b):
            if concept not in concepts:
                return result(False, f"{concept!r} is not a concept of {arguments.dataset.value}")
        submitted.append((min(pair.a, pair.b), max(pair.a, pair.b)))
    if len(set(submitted)) != len(submitted):
        return result(False, "a pair is submitted twice, counting (a, b) and (b, a) as one pair")

    # 2. Budget: the ledger is this task's, every charge is rederived, and the total fits the cost budget.
    ledger = read_json(Path(workspace) / LEDGER)
    if ledger is None and submitted:
        return result(False, f"there is no {LEDGER}, so no submitted pair was checked")
    related = {(pair.a, pair.b): pair for pair in reference.related}
    calls = [] if ledger is None else ledger["checks"]
    used = 0
    if ledger is not None:
        problem = locked_problem(ledger, arguments)
        if problem:
            return result(False, f"{problem} than the task")
        if ledger["frequent_postings"] != reference.frequent_postings:
            return result(False, f"the ledger charges signatures for P = {ledger['frequent_postings']}, not {reference.frequent_postings}")
        problem = charges_problem(ledger)
        if problem:
            return result(False, problem)
        for call in ledger["signatures"]:
            saved = read_json(Path(workspace) / CANDIDATES / f"{call['signature_id']}.json")
            if saved is None:
                return result(False, f"signatures {call['signature_id']}'s ranked pairs are not saved in the workspace")
            ranked = {(a, b): cost for _, a, b, cost in saved["pairs"]}
            expected = {key: pair.intersection + pair.union for key, pair in related.items()}
            if ranked != expected:
                return result(False, f"signatures {call['signature_id']} do not rank exactly the related pairs at their cost")
        checked = [tuple(pair) for call in calls for pair in call["checked"]]
        if len(set(checked)) != len(checked):
            return result(False, "the ledger checks a pair twice")
        for call in calls:
            pairs = [tuple(pair) for pair in call["checked"]]
            if any(pair not in related for pair in pairs):
                return result(False, "the ledger checks a pair that is not related")
            cost = sum(related[pair].intersection + related[pair].union for pair in pairs)
            if call["charged"] != cost:
                return result(False, f"the ledger charges a check {call['charged']}, not its |A| + |B| total {cost}")
        used = cost_used(ledger)
    if used > arguments.cost_budget:
        return result(False, f"signatures and checks cost {used}, over the cost budget of {arguments.cost_budget}")

    # 3. Precision: every submitted pair is a gap by the reference.
    gaps = {(pair.a, pair.b) for pair in reference.gaps}
    false = [pair for pair in submitted if pair not in gaps]
    if false:
        return result(False, f"{len(false)} submitted pairs are not gaps, e.g. {false[0]}", false_pairs=len(false))

    # 4. Provenance: check_pairs returned every submitted pair as a gap, with the reference's exact sizes.
    returned = {(gap["a"], gap["b"]): (gap["intersection"], gap["union"]) for call in calls for gap in call["gaps"]}
    for pair in submitted:
        if pair not in returned:
            return result(False, f"pair {pair} was never returned as a gap by check_pairs")
        truth = (related[pair].intersection, related[pair].union)
        if returned[pair] != truth:
            return result(False, f"pair {pair} was recorded with sizes {returned[pair]}, not its exact {truth}")

    # 5. Recall.
    recall = Fraction(len(submitted), len(gaps)) if gaps else Fraction(1)
    checked = sum(len(call["checked"]) for call in calls)
    details = {"submitted": len(submitted), "reference": len(gaps), "cost": used, "checked": checked}
    message = f"found {len(submitted)} of {len(gaps)} gaps, spending {used} of {arguments.cost_budget}"
    if recall < min_recall:
        return result(False, f"{message}: recall {float(recall):.3f} is below {arguments.min_recall}", score=float(recall), **details)
    return result(True, message, score=float(recall), **details)
