from .checks import decimals_problem, result
from .rule import is_gap, nested, parse_decimal


def verify_input(value):
    problem = decimals_problem(
        threshold=value.threshold, max_jaccard=value.max_jaccard, min_recall=value.min_recall
    )
    if problem:
        return result(False, problem)
    if value.min_papers < 1 or value.cost_budget < 1:
        return result(False, "min_papers and cost_budget must each be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    out = output_value
    if (out.dataset, out.min_papers) != (input_value.dataset, input_value.min_papers) or (
        parse_decimal(out.threshold), parse_decimal(out.max_jaccard)
    ) != (parse_decimal(input_value.threshold), parse_decimal(input_value.max_jaccard)):
        return result(False, "output is for another dataset, min_papers, threshold or max_jaccard than the input")
    concepts = set(out.concepts)
    if len(concepts) != len(out.concepts) or len(concepts) < 2:
        return result(False, "the concepts must be at least 2 and unique")
    pairs = [(pair.a, pair.b) for pair in out.related]
    if pairs != sorted(set(pairs)):
        return result(False, "the related pairs must be unique and sorted by a then b")
    for pair in out.related:
        if pair.a not in concepts or pair.b not in concepts or not pair.a < pair.b:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) is not two known concepts in canonical order")
        if nested(pair.a, pair.b):
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) is nested, so never related")
        if not 0 <= pair.intersection <= pair.union or pair.intersection + pair.union < 2 * out.min_papers:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) has sizes impossible for two frequent concepts")
    max_jaccard = parse_decimal(input_value.max_jaccard)
    expected = [(p.a, p.b) for p in out.related if is_gap(p.intersection, p.union, max_jaccard)]
    if [(pair.a, pair.b) for pair in out.gaps] != expected:
        return result(False, "the gaps are not exactly the related pairs with Jaccard at most max_jaccard")
    if out.frequent_postings < 0:
        return result(False, "P cannot be negative")
    return result(True, "every related pair and gap is listed", related=len(pairs), gaps=len(expected))
