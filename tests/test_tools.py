from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
import verifier.rule
from contract import TOOLS, CheckInput, Dataset, DescribeInput, FindInput, SignaturesInput, Tool
from plugin import build_signatures, check_pairs, describe_corpus, exact_gaps, rule
from plugin.bundle import LEDGER, hash_rows, load

ROOT = Path(__file__).resolve().parents[1]
EDGE = {"dataset": Dataset.EDGE_CASES, "min_papers": 4, "threshold": "0.80", "max_jaccard": "0.25"}


def checked(tool: Tool, given, made, workspace: Path) -> dict:
    """verify_output on a tool call, with values built the way the sandbox builds them."""
    given_class, made_class = TOOLS[tool]
    return verifier.verify_output(
        tool.value,
        build_output(plain_output(given), given_class),
        build_output(plain_output(made), made_class),
        str(workspace),
    )


def signatures(workspace: Path, rows: int, cost_budget: int = 60, **changes):
    given = SignaturesInput(rows=rows, cost_budget=cost_budget, **(EDGE | changes))
    return given, build_signatures(given, workspace)


def check(workspace: Path, start: int, number: int, cost_budget: int = 60, signature_id: str = "edge-cases-r8", **changes):
    fields = {"signature_id": signature_id, "start_rank": start, "count": number, "cost_budget": cost_budget}
    given = CheckInput(**(EDGE | fields | changes))
    return given, check_pairs(given, workspace)


def ranked(workspace: Path, signature_id: str) -> list[list]:
    return json.loads((workspace / "candidates" / f"{signature_id}.json").read_text())["pairs"]


def slow_gaps(dataset: Dataset, min_papers: int, threshold: str, max_jaccard: str) -> dict[tuple[str, str], tuple]:
    """A second exact computation, straight from the raw JSON: related pairs by name with (intersection, union, gap).
    It uses the bundled nested_pairs list rather than rule.nested, and set operations on names."""
    raw = json.loads((ROOT / "src/plugin/data" / f"{dataset.value}.json").read_text())
    concepts = raw["concepts"]
    nested = {frozenset(pair) for pair in raw["nested_pairs"]}
    t, ceiling = Fraction(threshold), Fraction(max_jaccard)
    found = {}
    for i, first in enumerate(concepts):
        for j, second in enumerate(concepts):
            if j <= i or len(first["papers"]) < min_papers or len(second["papers"]) < min_papers:
                continue
            if frozenset((i, j)) in nested:
                continue
            dot = sum(u * v for u, v in zip(first["embedding"], second["embedding"]))
            norms = sum(u * u for u in first["embedding"]) * sum(v * v for v in second["embedding"])
            if dot < 0 or Fraction(dot * dot, norms) < t * t:
                continue
            a, b = set(first["papers"]), set(second["papers"])
            key = tuple(sorted((first["name"], second["name"])))
            found[key] = (len(a & b), len(a | b), Fraction(len(a & b), len(a | b)) <= ceiling)
    return found


@pytest.mark.parametrize(
    ("dataset", "min_papers", "threshold", "sizes"),
    [
        (Dataset.ACL_SMALL, 10, "0.70", (2924, 493, 124, 7342, 42, 6750)),
        (Dataset.ACL_FULL, 50, "0.70", (21934, 493, 181, 55767, 69, 53234)),
        (Dataset.EDGE_CASES, 4, "0.80", (76, 18, 17, 87, 6, 65)),
    ],
)
def test_describe_corpus_counts_the_bundle(tmp_path: Path, dataset: Dataset, min_papers: int, threshold: str, sizes) -> None:
    given = DescribeInput(dataset=dataset, min_papers=min_papers, threshold=threshold)
    made = describe_corpus(given, tmp_path)
    assert (
        made.papers, made.concepts, made.frequent_concepts, made.frequent_postings, made.related_pairs,
        made.related_check_cost,
    ) == sizes
    assert (made.hash_rows, made.row_rate) == (len(hash_rows()[1]), "0.001")
    assert checked(Tool.DESCRIBE_CORPUS, given, made, tmp_path)["valid"] is True
    assert not (tmp_path / LEDGER).exists()  # describing charges nothing


def test_the_nested_rule_matches_the_bundled_list() -> None:
    for dataset in Dataset:
        raw = json.loads((ROOT / "src/plugin/data" / f"{dataset.value}.json").read_text())
        names = [concept["name"] for concept in raw["concepts"]]
        found = [[i, j] for i in range(len(names)) for j in range(i + 1, len(names)) if rule.nested(names[i], names[j])]
        assert found == raw["nested_pairs"]
    assert rule.nested("translation", "machine translation") and not rule.nested("at-floor", "below-floor")


def test_minhash_rows_match_as_jaccard_says(tmp_path: Path) -> None:
    signatures(tmp_path, 256)
    matches = {(a, b): m for m, a, b, _ in ranked(tmp_path, "edge-cases-r256")}
    assert matches[("apart-a", "apart-b")] == 0  # never together: Jaccard 0, no row ever matches
    assert matches[("at-floor", "floor-mate")] == 0
    assert matches[("twin-a", "twin-b")] == 256  # the same papers: every row matches
    prime, rows = hash_rows()
    corpus = load(Dataset.EDGE_CASES)
    papers = dict(zip(corpus.names, corpus.papers))
    for (a, b), m in matches.items():  # recount from the hash rows directly
        assert m == sum(min((x * p + y) % prime for p in papers[a]) == min((x * p + y) % prime for p in papers[b])
                        for x, y in rows)


def test_a_shorter_signature_is_the_start_of_a_longer_one(tmp_path: Path) -> None:
    signatures(tmp_path / "long", 256)
    signatures(tmp_path / "short", 16)
    prime, rows = hash_rows()
    corpus = load(Dataset.EDGE_CASES)
    papers = dict(zip(corpus.names, corpus.papers))
    short = {(a, b): m for m, a, b, _ in ranked(tmp_path / "short", "edge-cases-r16")}
    for (a, b), m in short.items():
        first16 = [min((x * p + y) % prime for p in papers[a]) == min((x * p + y) % prime for p in papers[b])
                   for x, y in rows[:16]]
        assert m == sum(first16)


def test_build_signatures_is_deterministic_and_ranked(tmp_path: Path) -> None:
    given = SignaturesInput(Dataset.ACL_FULL, 64, 50, "0.70", "0.005", 24000)
    first = build_signatures(given, tmp_path / "one")
    second = build_signatures(given, tmp_path / "two")
    assert first == second and first.signature_id == "acl-full-r64" and first.related_pairs == 69
    assert first.cost_charged == 3570  # ceil(64 * 55767 / 1000)
    saved = (tmp_path / "one" / "candidates" / "acl-full-r64.json").read_text()
    assert saved == (tmp_path / "two" / "candidates" / "acl-full-r64.json").read_text()
    assert sum(row.pairs for row in first.histogram) == first.related_pairs
    assert sum(row.check_cost for row in first.histogram) == 53234
    assert first.histogram[0].first_rank == 1 and first.histogram[-1].last_rank == first.related_pairs
    assert checked(Tool.BUILD_SIGNATURES, given, first, tmp_path / "one")["valid"] is True
    run = json.loads(saved)
    run["pairs"].reverse()
    (tmp_path / "one" / "candidates" / "acl-full-r64.json").write_text(json.dumps(run))
    assert checked(Tool.BUILD_SIGNATURES, given, first, tmp_path / "one")["valid"] is False


def test_signatures_are_charged_only_for_new_rows(tmp_path: Path) -> None:
    _, made = signatures(tmp_path, 8)
    assert (made.cost_charged, made.cost_used, made.cost_remaining) == (1, 1, 59)  # ceil(8 * 87 / 1000)
    given, made = signatures(tmp_path, 8)  # the same signatures again
    assert (made.cost_charged, made.cost_used) == (0, 1)
    assert checked(Tool.BUILD_SIGNATURES, given, made, tmp_path)["valid"] is True
    given, made = signatures(tmp_path, 4)  # fewer rows: the first 4 rows are built already
    assert (made.cost_charged, made.cost_used) == (0, 1)
    assert checked(Tool.BUILD_SIGNATURES, given, made, tmp_path)["valid"] is True
    given, made = signatures(tmp_path, 256)  # ceil(256 * 87 / 1000) - ceil(8 * 87 / 1000)
    assert (made.cost_charged, made.cost_used, made.cost_remaining) == (22, 23, 37)
    assert checked(Tool.BUILD_SIGNATURES, given, made, tmp_path)["valid"] is True
    made.cost_charged = 23
    assert checked(Tool.BUILD_SIGNATURES, given, made, tmp_path)["valid"] is False


def test_extending_signatures_on_acl_full_charges_the_difference(tmp_path: Path) -> None:
    given = SignaturesInput(Dataset.ACL_FULL, 64, 50, "0.70", "0.005", 24000)
    assert build_signatures(given, tmp_path).cost_charged == 3570
    given.rows = 128
    made = build_signatures(given, tmp_path)
    assert (made.cost_charged, made.cost_used) == (7139 - 3570, 7139)
    assert checked(Tool.BUILD_SIGNATURES, given, made, tmp_path)["valid"] is True


def test_signatures_alone_can_exceed_the_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"256-row signatures cost 23 more .* only 20 of the cost budget 20 remain"
                                         r".* rows may be at most 229"):
        signatures(tmp_path, 256, cost_budget=20)
    assert not (tmp_path / LEDGER).exists()
    signatures(tmp_path, 200, cost_budget=20)  # ceil(200 * 87 / 1000) = 18
    with pytest.raises(ValueError, match="rows may be at most 229; up to 200 rows cost nothing more"):
        signatures(tmp_path, 256, cost_budget=20)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rows": 0}, "rows must be from 1 to 256"),
        ({"rows": 257}, "rows must be from 1 to 256"),
        ({"min_papers": 0}, "min_papers must be at least 1"),
        ({"cost_budget": 0}, "cost_budget must be at least 1"),
        ({"threshold": "0.8e0"}, "threshold must be a decimal string"),
        ({"max_jaccard": "0"}, "max_jaccard must be a decimal string"),
    ],
)
def test_build_signatures_says_what_is_wrong(tmp_path: Path, changes: dict, message: str) -> None:
    fields = {"rows": 8} | changes
    with pytest.raises(ValueError, match=message):
        signatures(tmp_path, **fields)


def test_the_ledger_keeps_the_task_fixed(tmp_path: Path) -> None:
    signatures(tmp_path, 8)
    signatures(tmp_path, 8, threshold="0.8")  # the same value
    with pytest.raises(ValueError, match="pass cost_budget 60"):
        signatures(tmp_path, 8, cost_budget=61)
    with pytest.raises(ValueError, match='pass threshold "0.80"'):
        signatures(tmp_path, 8, threshold="0.75")
    with pytest.raises(ValueError, match='pass max_jaccard "0.25"'):
        check(tmp_path, 1, 1, max_jaccard="0.3")
    with pytest.raises(ValueError, match="pass min_papers 4"):
        check(tmp_path, 1, 1, min_papers=5)


@pytest.fixture
def built(tmp_path: Path) -> Path:
    signatures(tmp_path, 8)
    return tmp_path


def test_check_pairs_measures_exactly_and_charges_only_new_pairs(built: Path) -> None:
    pairs = ranked(built, "edge-cases-r8")
    given, made = check(built, 1, 2)
    assert (made.checked, made.skipped, made.cost_charged) == (2, 0, pairs[0][3] + pairs[1][3])
    assert made.cost_remaining == 59 - made.cost_charged
    assert checked(Tool.CHECK_PAIRS, given, made, built)["valid"] is True
    before = made.cost_used
    given, made = check(built, 2, 2)
    assert (made.checked, made.skipped, made.cost_charged, made.total_checked) == (1, 1, pairs[2][3], 3)
    assert made.cost_used == before + pairs[2][3]
    assert checked(Tool.CHECK_PAIRS, given, made, built)["valid"] is True


def test_a_pair_checked_under_one_signature_is_skipped_under_another(tmp_path: Path) -> None:
    signatures(tmp_path, 8, cost_budget=100)
    check(tmp_path, 1, 6, cost_budget=100)
    signatures(tmp_path, 4, cost_budget=100)
    given, made = check(tmp_path, 1, 6, cost_budget=100, signature_id="edge-cases-r4")
    assert (made.checked, made.skipped, made.cost_charged, made.gaps) == (0, 6, 0, [])
    assert checked(Tool.CHECK_PAIRS, given, made, tmp_path)["valid"] is True


def test_check_pairs_refuses_a_range_over_budget_and_says_what_fits(built: Path) -> None:
    costs = [cost for *_, cost in ranked(built, "edge-cases-r8")]
    fits = max(n for n in range(7) if sum(costs[:n]) <= 59)
    with pytest.raises(ValueError, match=rf"ranks 1 to 6 hold 6 pairs not yet checked \(0 already checked\), costing 65,"
                                         rf" but only 59 of the cost budget 60 remain: with start_rank 1, count may be at most {fits}"):
        check(built, 1, 6)
    assert json.loads((built / LEDGER).read_text())["checks"] == []


def test_check_pairs_says_when_one_pair_alone_is_too_costly(tmp_path: Path) -> None:
    signatures(tmp_path, 8, cost_budget=9)  # 1 for the signatures, 8 left; every pair costs at least 9
    costs = [cost for *_, cost in ranked(tmp_path, "edge-cases-r8")]
    with pytest.raises(ValueError, match=rf"the pair at rank 1 alone costs {costs[0]}, so no range from rank 1 fits"):
        check(tmp_path, 1, 1, cost_budget=9)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"signature_id": "edge-cases-r9"}, "call build_signatures for them first"),
        ({"signature_id": "acl-small-r8"}, "is not a edge-cases signature"),
        ({"signature_id": "../x"}, "is not a edge-cases signature"),
        ({"start_rank": 0}, "at least 1"),
        ({"cost_budget": 0}, "cost_budget must be at least 1"),
        ({"start_rank": 100}, "start_rank must be at most 6"),
        ({"count": 100}, "count may be at most 6"),
    ],
)
def test_check_pairs_says_what_is_wrong(built: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check(built, 1, 1, **changes)


def test_check_pairs_needs_signatures(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="call build_signatures first"):
        check(tmp_path, 1, 1)


def test_verify_output_refuses_a_tampered_check(built: Path) -> None:
    given, made = check(built, 1, 5)
    assert made.gaps
    assert checked(Tool.CHECK_PAIRS, given, made, built)["valid"] is True
    made.cost_remaining += 1
    assert checked(Tool.CHECK_PAIRS, given, made, built)["valid"] is False
    made.cost_remaining -= 1
    made.gaps[0].union += 1
    assert checked(Tool.CHECK_PAIRS, given, made, built)["valid"] is False


@pytest.mark.parametrize(
    ("dataset", "min_papers", "threshold", "max_jaccard", "related", "gaps"),
    [
        (Dataset.ACL_SMALL, 10, "0.70", "0.01", 42, 17),
        (Dataset.ACL_SMALL, 50, "0.70", "0.005", 7, 0),
        (Dataset.ACL_FULL, 50, "0.70", "0.005", 69, 23),
        (Dataset.EDGE_CASES, 4, "0.80", "0.25", 6, 3),
    ],
)
def test_the_reference_matches_a_slow_computation(
    tmp_path: Path, dataset: Dataset, min_papers: int, threshold: str, max_jaccard: str, related: int, gaps: int
) -> None:
    given = FindInput(dataset, min_papers, threshold, max_jaccard, 1, "1")
    found = exact_gaps(given, tmp_path)
    slow = slow_gaps(dataset, min_papers, threshold, max_jaccard)
    assert {(p.a, p.b): (p.intersection, p.union) for p in found.related} == {k: v[:2] for k, v in slow.items()}
    assert [(p.a, p.b) for p in found.gaps] == sorted(k for k, v in slow.items() if v[2])
    assert (len(found.related), len(found.gaps)) == (related, gaps)
    assert checked(Tool.EXACT_GAPS, given, found, tmp_path)["valid"] is True


def test_edge_cases_hold_the_boundaries() -> None:
    slow = slow_gaps(Dataset.EDGE_CASES, 4, "0.80", "0.25")
    assert slow[("ceiling-a", "ceiling-b")] == (2, 8, True)  # Jaccard exactly max_jaccard, cosine exactly 0.80
    assert slow[("above-a", "above-b")] == (3, 11, False)  # just above
    assert slow[("apart-a", "apart-b")] == (0, 9, True)  # never together
    assert slow[("at-floor", "floor-mate")] == (0, 10, True)  # at-floor holds exactly min_papers papers
    assert ("below-floor", "below-mate") not in slow  # below-floor holds min_papers - 1
    assert ("machine translation", "translation") not in slow  # nested, though similar and frequent
    assert ("unlike-a", "unlike-b") not in slow  # cosine 0.60
    assert slow[("twin-a", "twin-b")] == (5, 5, False)
    corpus = load(Dataset.EDGE_CASES)
    sizes = dict(zip(corpus.names, map(len, corpus.papers)))
    assert (sizes["at-floor"], sizes["below-floor"]) == (4, 3)
    assert ("below-floor", "below-mate") in slow_gaps(Dataset.EDGE_CASES, 3, "0.80", "0.25")


def test_the_two_rule_files_are_identical() -> None:
    assert (ROOT / "src/plugin/rule.py").read_text() == (ROOT / "verifier/rule.py").read_text()


def test_the_histogram_rule() -> None:
    assert rule.histogram([(0, 5), (0, 7), (1, 4), (3, 9)], 8) == [
        (0, 0, "0.0000", "0.0000", 2, 12, 1, 2), (1, 1, "0.1250", "0.1250", 1, 4, 3, 3), (3, 3, "0.3750", "0.3750", 1, 9, 4, 4)
    ]
    ranked_pairs = [(m, 1) for m in range(20)]  # 20 distinct values: ranges 0, 1, 2-3, 4-7, 8-15, 16-31
    buckets = rule.histogram(ranked_pairs, 256)
    assert [(low, high, pairs) for low, high, _, _, pairs, *_ in buckets] == [
        (0, 0, 1), (1, 1, 1), (2, 3, 2), (4, 7, 4), (8, 15, 8), (16, 19, 4)
    ]
    assert buckets[-1][7] == 20 and rule.histogram([], 8) == []
    assert verifier.rule.histogram(ranked_pairs, 256) == buckets
    assert rule.estimate(1, 3) == "0.3333" and rule.estimate(3, 3) == "1.0000" and rule.estimate(1, 256) == "0.0039"


def test_the_cost_rules() -> None:
    assert rule.signature_cost(64, 55767) == 3570 and rule.signature_cost(0, 55767) == 0
    assert rule.signature_charge(128, 64, 55767) == 7139 - 3570
    assert rule.signature_charge(64, 128, 55767) == 0 and rule.signature_charge(64, 64, 55767) == 0
    assert rule.is_gap(1, 4, Fraction(1, 4)) and not rule.is_gap(3, 11, Fraction(1, 4)) and rule.is_gap(0, 9, Fraction(1, 1000))


@pytest.mark.parametrize("text", ["0", "0.0", "1.001", "1.5", "0.8000", ".8", "0.8e0", " 0.8", "-0.5", "0,8", "nan", ""])
def test_decimals_outside_the_rule_are_refused(text: str) -> None:
    assert rule.parse_decimal(text) is None and verifier.rule.parse_decimal(text) is None


def solve(workspace: Path, dataset: Dataset, min_papers: int, threshold: str, max_jaccard: str, budget: int, rows: int):
    """A single signature, then the longest prefix of the ranking that fits the remaining budget, in one check."""
    made = build_signatures(SignaturesInput(dataset, rows, min_papers, threshold, max_jaccard, budget), workspace)
    costs = [cost for *_, cost in ranked(workspace, made.signature_id)]
    count, spent = 0, 0
    while count < len(costs) and spent + costs[count] <= made.cost_remaining:
        spent += costs[count]
        count += 1
    if not count:
        return []
    found = check_pairs(CheckInput(dataset, made.signature_id, 1, count, min_papers, threshold, max_jaccard, budget), workspace)
    return found.gaps


@pytest.mark.parametrize(
    ("dataset", "min_papers", "max_jaccard", "budget", "rows", "found"),
    [
        (Dataset.ACL_FULL, 50, "0.005", 24000, 32, 14),
        (Dataset.ACL_FULL, 50, "0.005", 24000, 64, 23),
        (Dataset.ACL_FULL, 50, "0.005", 24000, 128, 21),
        (Dataset.ACL_FULL, 50, "0.005", 24000, 256, 19),
        (Dataset.ACL_SMALL, 10, "0.01", 3800, 32, 14),
        (Dataset.ACL_SMALL, 10, "0.01", 3800, 64, 16),
        (Dataset.ACL_SMALL, 10, "0.01", 3800, 128, 17),
        (Dataset.ACL_SMALL, 10, "0.01", 3800, 256, 15),
    ],
)
def test_the_instances_separate_settings(
    tmp_path: Path, dataset: Dataset, min_papers: int, max_jaccard: str, budget: int, rows: int, found: int
) -> None:
    """Checking the longest affordable prefix of one signature finds these many gaps, as measured at checkpoint 2."""
    assert len(solve(tmp_path, dataset, min_papers, "0.70", max_jaccard, budget, rows)) == found
