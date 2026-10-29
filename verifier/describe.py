from .checks import decimals_problem, result
from .rule import HASH_ROWS, ROW_RATE


def verify_input(value):
    problem = decimals_problem(threshold=value.threshold)
    if problem:
        return result(False, problem)
    if value.min_papers < 1:
        return result(False, "min_papers must be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    out = output_value
    if out.dataset != input_value.dataset:
        return result(False, f"output describes {out.dataset.value}, not {input_value.dataset.value}")
    if out.papers < 1 or not 0 <= out.frequent_concepts <= out.concepts:
        return result(False, "a corpus needs a paper, and its frequent concepts are some of its concepts")
    if out.frequent_postings < out.frequent_concepts * input_value.min_papers:
        return result(False, "every frequent concept holds at least min_papers papers, so P cannot be smaller")
    if not 0 <= out.related_pairs <= out.frequent_concepts * (out.frequent_concepts - 1) // 2:
        return result(False, f"{out.frequent_concepts} frequent concepts make at most F choose 2 related pairs")
    if out.related_check_cost < 2 * input_value.min_papers * out.related_pairs:
        return result(False, "every related pair costs at least 2 * min_papers to check")
    if (out.hash_rows, out.row_rate) != (HASH_ROWS, ROW_RATE):
        return result(False, f"the plugin bundles {HASH_ROWS} hash rows at rate {ROW_RATE}")
    return result(True, "the corpus is described", frequent=out.frequent_concepts, related=out.related_pairs)
