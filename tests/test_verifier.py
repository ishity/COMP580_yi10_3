from __future__ import annotations

import json
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import TASKS, TOOLS, CheckInput, Dataset, ExactOutput, FindInput, SignaturesInput, Task, Tool
from plugin import build_signatures, check_pairs, exact_gaps
from plugin.bundle import ledger_key

EDGE = {"dataset": "edge-cases", "min_papers": 4, "threshold": "0.80", "max_jaccard": "0.25", "cost_budget": 60,
        "min_recall": "1.0"}
EMPTY = {"dataset": "acl-small", "min_papers": 50, "threshold": "0.70", "max_jaccard": "0.005", "cost_budget": 1000,
         "min_recall": "1.0"}
FULL = {"dataset": "acl-full", "min_papers": 50, "threshold": "0.70", "max_jaccard": "0.005", "cost_budget": 24000,
        "min_recall": "0.90"}
SMALL = {"dataset": "acl-small", "min_papers": 10, "threshold": "0.70", "max_jaccard": "0.01", "cost_budget": 3800,
         "min_recall": "0.90"}
AT_CEILING = {"a": "ceiling-a", "b": "ceiling-b"}  # Jaccard 2/8, exactly max_jaccard


# The verifier gets contract instances, as NanoAgent's sandbox builds them from JSON.
def verify_task(arguments: dict, pairs: list[dict], workspace: Path, task: dict | None = None) -> dict:
    """The verdict on ``pairs`` for the task ``arguments``, whose reference exact_gaps computes from ``task`` (the
    same arguments unless given)."""
    given, made = TASKS[Task.FIND_CONCEPT_GAPS]
    reference = plain_output(exact_gaps(build_output(task or arguments, given), workspace / "scratch"))
    return verifier.verify_task(
        Task.FIND_CONCEPT_GAPS.value,
        build_output(arguments, given),
        build_output({"pairs": pairs}, made),
        build_output(reference, ExactOutput),
        str(workspace),
    )


def verify_input(tool: Tool, value: dict) -> dict:
    return verifier.verify_input(tool.value, build_output(value, TOOLS[tool][0]))


def reference(arguments: dict) -> list[dict]:
    found = exact_gaps(build_output(arguments, FindInput), Path("."))
    return [{"a": pair.a, "b": pair.b} for pair in found.gaps]


def key(arguments: dict) -> str:
    """The name of the ledger for the task ``arguments``."""
    return ledger_key(build_output(arguments, FindInput))


def ledger_file(workspace: Path, arguments: dict) -> Path:
    return workspace / "ledgers" / f"{key(arguments)}.json"


def ranking_file(workspace: Path, arguments: dict, signature_id: str) -> Path:
    return workspace / "candidates" / key(arguments) / f"{signature_id}.json"


def task_values(arguments: dict) -> tuple:
    dataset = Dataset(arguments["dataset"])
    return dataset, arguments["min_papers"], arguments["threshold"], arguments["max_jaccard"], arguments["cost_budget"]


def solved(workspace: Path, arguments: dict, rows: int, last_rank: int | None = None) -> list[dict]:
    """Build one signature, check ranks 1 to ``last_rank`` (the longest prefix that fits when None) and return the gaps."""
    dataset, min_papers, threshold, max_jaccard, budget = task_values(arguments)
    made = build_signatures(SignaturesInput(dataset, rows, min_papers, threshold, max_jaccard, budget), workspace)
    if last_rank is None:
        saved = json.loads(ranking_file(workspace, arguments, made.signature_id).read_text())["pairs"]
        last_rank, spent = 0, 0
        while last_rank < len(saved) and spent + saved[last_rank][3] <= made.cost_remaining:
            spent += saved[last_rank][3]
            last_rank += 1
    if not last_rank:
        return []
    found = check_pairs(
        CheckInput(dataset, made.signature_id, 1, last_rank, min_papers, threshold, max_jaccard, budget), workspace
    )
    return [{"a": pair.a, "b": pair.b} for pair in found.gaps]


def edit_ledger(workspace: Path, change, arguments: dict = EDGE) -> None:
    ledger = json.loads(ledger_file(workspace, arguments).read_text())
    change(ledger)
    ledger_file(workspace, arguments).write_text(json.dumps(ledger))


def relabel(workspace: Path, arguments: dict, changes: dict) -> None:
    """Move the ledger and rankings of ``arguments`` to the task values ``arguments | changes``, recording those values,
    as if that other task had spent what this one did."""
    edit_ledger(workspace, lambda ledger: ledger.update(changes), arguments)
    ledger_file(workspace, arguments).rename(ledger_file(workspace, arguments | changes))
    (workspace / "candidates" / key(arguments)).rename(workspace / "candidates" / key(arguments | changes))


@pytest.fixture
def edge(tmp_path: Path) -> list[dict]:
    """The longest affordable prefix of an 8-row edge-cases signature checked in ``tmp_path``; the gaps found."""
    found = solved(tmp_path, EDGE, 8)
    assert sorted(found, key=lambda pair: (pair["a"], pair["b"])) == reference(EDGE)
    return found


def test_approves_every_gap(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0
    assert verdict["details"] == {"submitted": 3, "reference": 3, "cost": 56, "checked": 5}


def test_the_verdict_is_the_same_twice(tmp_path: Path, edge: list[dict]) -> None:
    assert verify_task(EDGE, edge, tmp_path) == verify_task(EDGE, edge, tmp_path)


def test_a_pair_at_exactly_max_jaccard_is_a_gap_and_one_just_above_is_not(tmp_path: Path, edge: list[dict]) -> None:
    assert AT_CEILING in edge
    assert verify_task(EDGE | {"min_recall": "0.333"}, [AT_CEILING], tmp_path)["valid"] is True
    verdict = verify_task(EDGE, [*edge, {"a": "above-a", "b": "above-b"}], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "are not gaps" in verdict["message"]


def test_a_pair_that_never_co_occurs_is_a_gap(tmp_path: Path, edge: list[dict]) -> None:
    ledger = json.loads(ledger_file(tmp_path, EDGE).read_text())
    gaps = {(g["a"], g["b"]): (g["intersection"], g["union"]) for g in ledger["checks"][0]["gaps"]}
    assert gaps[("apart-a", "apart-b")] == (0, 9)
    saved = json.loads(ranking_file(tmp_path, EDGE, "edge-cases-r8").read_text())["pairs"]
    assert [matches for matches, a, *_ in saved if a == "apart-a"] == [0]  # its rows never match


@pytest.mark.parametrize(
    "pair",
    [
        {"a": "below-floor", "b": "below-mate"},  # below-floor holds min_papers - 1 papers
        {"a": "machine translation", "b": "translation"},  # nested, though similar, frequent and never together
        {"a": "unlike-a", "b": "unlike-b"},  # cosine 0.60, below threshold
        {"a": "twin-a", "b": "twin-b"},  # Jaccard 1
    ],
    ids=["below-floor", "nested", "dissimilar", "twin"],
)
def test_refuses_a_pair_that_is_not_a_gap(tmp_path: Path, edge: list[dict], pair: dict) -> None:
    verdict = verify_task(EDGE, [*edge, pair], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "are not gaps" in verdict["message"]


def test_a_concept_at_exactly_min_papers_forms_a_gap(tmp_path: Path, edge: list[dict]) -> None:
    assert {"a": "at-floor", "b": "floor-mate"} in edge
    assert verify_task(EDGE, edge, tmp_path)["valid"] is True


def test_refuses_an_empty_answer_when_gaps_exist(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE, [], tmp_path)
    assert verdict["valid"] is False and verdict["score"] == 0.0 and "recall 0.000" in verdict["message"]


def test_approves_an_empty_answer_when_no_pair_is_a_gap(tmp_path: Path) -> None:
    assert reference(EMPTY) == []  # the lowest related Jaccard is 0.0053, just above 0.005
    assert verify_task(EMPTY, [], tmp_path)["valid"] is True  # no ledger: nothing was checked
    assert solved(tmp_path, EMPTY, 64) == []
    verdict = verify_task(EMPTY, [], tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0 and verdict["details"]["checked"] > 0


def test_accepts_a_reversed_pair_but_not_a_repeated_one(tmp_path: Path, edge: list[dict]) -> None:
    reversed_pairs = [{"a": pair["b"], "b": pair["a"]} for pair in edge]
    assert verify_task(EDGE, reversed_pairs, tmp_path)["valid"] is True
    for repeat in (edge[0], reversed_pairs[0]):
        verdict = verify_task(EDGE, [*edge, repeat], tmp_path)
        assert verdict["valid"] is False and "submitted twice" in verdict["message"]


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        ({"a": "ceiling-a", "b": "no such concept"}, "is not a concept of edge-cases"),
        ({"a": "ceiling-a", "b": "ceiling-a"}, "with itself"),
    ],
)
def test_refuses_a_malformed_pair(tmp_path: Path, edge: list[dict], pair: dict, message: str) -> None:
    verdict = verify_task(EDGE, [*edge, pair], tmp_path)
    assert verdict["valid"] is False and message in verdict["message"]


def test_refuses_signatures_alone_over_the_budget(tmp_path: Path) -> None:
    solved(tmp_path, EDGE, 256, last_rank=0)  # 23 for 256 rows, nothing checked
    relabel(tmp_path, EDGE, {"cost_budget": 20})
    verdict = verify_task(EDGE | {"cost_budget": 20}, [], tmp_path, task=EDGE)
    assert verdict["valid"] is False and "cost 23, over the cost budget of 20" in verdict["message"]


def test_refuses_checks_over_the_budget(tmp_path: Path, edge: list[dict]) -> None:
    relabel(tmp_path, EDGE, {"cost_budget": 50})
    verdict = verify_task(EDGE | {"cost_budget": 50}, edge, tmp_path, task=EDGE)
    assert verdict["valid"] is False and "cost 56, over the cost budget of 50" in verdict["message"]


def test_rebuilding_the_same_signatures_is_not_charged_twice(tmp_path: Path, edge: list[dict]) -> None:
    solved(tmp_path, EDGE, 8, last_rank=1)  # the same 8 rows again, and rank 1 again
    ledger = json.loads(ledger_file(tmp_path, EDGE).read_text())
    assert [call["charged"] for call in ledger["signatures"]] == [1, 0]
    assert ledger["checks"][-1]["charged"] == 0
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is True and verdict["details"]["cost"] == 56


def test_refuses_a_rebuild_charged_again(tmp_path: Path, edge: list[dict]) -> None:
    solved(tmp_path, EDGE, 8, last_rank=1)
    edit_ledger(tmp_path, lambda ledger: ledger["signatures"][1].update(charged=1))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "charges signatures edge-cases-r8 1, not 0" in verdict["message"]


def test_refuses_signatures_charged_nothing_the_first_time(tmp_path: Path, edge: list[dict]) -> None:
    edit_ledger(tmp_path, lambda ledger: ledger["signatures"][0].update(charged=0))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "charges signatures edge-cases-r8 0, not 1" in verdict["message"]


def test_refuses_a_check_charged_less_than_its_pairs(tmp_path: Path, edge: list[dict]) -> None:
    edit_ledger(tmp_path, lambda ledger: ledger["checks"][0].update(charged=10))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "not its |A| + |B| total 55" in verdict["message"]


def test_refuses_a_ledger_with_another_postings_count(tmp_path: Path, edge: list[dict]) -> None:
    edit_ledger(tmp_path, lambda ledger: ledger.update(frequent_postings=10))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "for P = 10, not 87" in verdict["message"]


@pytest.mark.parametrize(
    "changes",
    [{"cost_budget": 61}, {"threshold": "0.75"}, {"max_jaccard": "0.3"}, {"min_papers": 3}],
    ids=["cost_budget", "threshold", "max_jaccard", "min_papers"],
)
def test_never_reads_the_ledger_of_another_task(tmp_path: Path, edge: list[dict], changes: dict) -> None:
    verdict = verify_task(EDGE | changes, edge, tmp_path)
    assert verdict["valid"] is False and f"there is no ledger ledgers/{key(EDGE | changes)}.json" in verdict["message"]


@pytest.mark.parametrize(
    "changes",
    [{"cost_budget": 61}, {"threshold": "0.75"}, {"max_jaccard": "0.3"}, {"min_papers": 3}, {"dataset": "acl-small"}],
    ids=["cost_budget", "threshold", "max_jaccard", "min_papers", "dataset"],
)
def test_refuses_a_ledger_whose_values_disagree_with_its_name(tmp_path: Path, edge: list[dict], changes: dict) -> None:
    edit_ledger(tmp_path, lambda ledger: ledger.update(changes))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "the ledger is for another" in verdict["message"]


def test_instances_sharing_a_workspace_never_lock_each_other(tmp_path: Path) -> None:
    """RUSH Lab keeps a member's workspace across runs: one instance's ledger must not lock or feed the next."""
    runs = [(EDGE, 8), (SMALL, 64), (EMPTY, 64), (FULL, 64)]  # SMALL and EMPTY share a dataset and a signature_id
    found = {}
    for arguments, rows in runs:
        found[key(arguments)] = solved(tmp_path, arguments, rows)
        assert verify_task(arguments, found[key(arguments)], tmp_path)["valid"] is True
    assert sorted(path.name for path in (tmp_path / "ledgers").iterdir()) == sorted(f"{key(a)}.json" for a, _ in runs)
    for arguments, _ in runs:  # every verdict still holds once all four have run
        verdict = verify_task(arguments, found[key(arguments)], tmp_path)
        assert verdict["valid"] is True and verdict["details"]["cost"] <= arguments["cost_budget"]
    assert len(found[key(EDGE)]) == 3 and found[key(EMPTY)] == []


def test_the_same_instance_run_twice_in_one_workspace_passes(tmp_path: Path, edge: list[dict]) -> None:
    """A second run of the instance finds the first run's ledger: rebuilding is free, and rechecking the same ranks
    returns the same gaps from the ledger, marked previously scored, so the second run's answer passes too."""
    assert verify_task(EDGE, edge, tmp_path)["valid"] is True
    dataset, min_papers, threshold, max_jaccard, budget = task_values(EDGE)
    for rows in (8, 4):  # the same signatures, or shorter ones: both built already
        made = build_signatures(SignaturesInput(dataset, rows, min_papers, threshold, max_jaccard, budget), tmp_path)
        assert (made.cost_charged, made.cost_used) == (0, 56)
        with pytest.raises(ValueError, match="only 4 of the cost budget 60 remain: with start_rank 1, count may be at most 5"):
            check_pairs(CheckInput(dataset, made.signature_id, 1, 6, min_papers, threshold, max_jaccard, budget), tmp_path)
        again = check_pairs(CheckInput(dataset, made.signature_id, 1, 5, min_papers, threshold, max_jaccard, budget), tmp_path)
        assert (again.checked, again.skipped, again.cost_charged) == (0, 5, 0)
        assert {(g.a, g.b) for g in again.gaps if g.previously_scored} == {(p["a"], p["b"]) for p in edge}
        rerun = [{"a": g.a, "b": g.b} for g in again.gaps]
        verdict = verify_task(EDGE, rerun, tmp_path)
        assert verdict["valid"] is True and verdict["score"] == 1.0 and verdict["details"]["cost"] <= 60


def test_a_decimal_written_another_way_is_the_same_task(tmp_path: Path, edge: list[dict]) -> None:
    assert verify_task(EDGE | {"threshold": "0.8", "max_jaccard": "0.250"}, edge, tmp_path)["valid"] is True


def test_refuses_a_ledger_that_checks_a_pair_twice(tmp_path: Path, edge: list[dict]) -> None:
    edit_ledger(tmp_path, lambda ledger: ledger["checks"].append(ledger["checks"][0] | {"gaps": []}))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "checks a pair twice" in verdict["message"]


def test_refuses_pairs_without_a_ledger(tmp_path: Path, edge: list[dict]) -> None:
    ledger_file(tmp_path, EDGE).unlink()
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "there is no ledger ledgers/edge-cases-m4-t0.800-j0.250-b60.json" in verdict["message"]


def test_refuses_a_gap_that_was_never_checked(tmp_path: Path) -> None:
    found = solved(tmp_path, EDGE, 8, last_rank=1)
    assert len(found) == 1
    verdict = verify_task(EDGE, reference(EDGE), tmp_path)
    assert verdict["valid"] is False and "never returned as a gap" in verdict["message"]
    assert verify_task(EDGE | {"min_recall": "0.333"}, found, tmp_path)["valid"] is True


def test_refuses_a_gap_recorded_with_other_sizes(tmp_path: Path, edge: list[dict]) -> None:
    edit_ledger(tmp_path, lambda ledger: ledger["checks"][0]["gaps"][0].update(union=99))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "not its exact" in verdict["message"]


def test_refuses_saved_signatures_that_hide_a_related_pair(tmp_path: Path, edge: list[dict]) -> None:
    path = ranking_file(tmp_path, EDGE, "edge-cases-r8")
    saved = json.loads(path.read_text())
    saved["pairs"].pop()
    path.write_text(json.dumps(saved))
    verdict = verify_task(EDGE, edge, tmp_path)
    assert verdict["valid"] is False and "do not rank exactly the related pairs" in verdict["message"]


def test_recall_exactly_at_min_recall_passes(tmp_path: Path, edge: list[dict]) -> None:
    verdict = verify_task(EDGE | {"min_recall": "0.666"}, edge[:2], tmp_path)
    assert verdict["valid"] is True and verdict["score"] == pytest.approx(2 / 3)
    verdict = verify_task(EDGE | {"min_recall": "0.667"}, edge[:2], tmp_path)
    assert verdict["valid"] is False and "below 0.667" in verdict["message"]


@pytest.mark.parametrize(
    ("arguments", "rows", "valid", "submitted"),
    [(FULL, 64, True, 23), (FULL, 128, True, 21), (FULL, 32, False, 14), (FULL, 256, False, 19),
     (SMALL, 64, True, 16), (SMALL, 128, True, 17), (SMALL, 32, False, 14), (SMALL, 256, False, 15)],
)
def test_the_main_instances_pass_good_signatures_and_fail_poor_ones(
    tmp_path: Path, arguments: dict, rows: int, valid: bool, submitted: int
) -> None:
    found = solved(tmp_path, arguments, rows)
    verdict = verify_task(arguments, found, tmp_path)
    assert verdict["valid"] is valid and verdict["details"]["submitted"] == submitted


@pytest.mark.parametrize(
    ("tool", "value", "valid"),
    [
        (Tool.EXACT_GAPS, EDGE, True),
        (Tool.EXACT_GAPS, EDGE | {"min_recall": "1"}, True),
        (Tool.EXACT_GAPS, EDGE | {"min_recall": "0.9e0"}, False),
        (Tool.EXACT_GAPS, EDGE | {"max_jaccard": "0"}, False),
        (Tool.EXACT_GAPS, EDGE | {"threshold": "1.5"}, False),
        (Tool.EXACT_GAPS, EDGE | {"min_papers": 0}, False),
        (Tool.EXACT_GAPS, EDGE | {"cost_budget": 0}, False),
        (Tool.BUILD_SIGNATURES, FULL | {"rows": 64}, True),
        (Tool.BUILD_SIGNATURES, FULL | {"rows": 0}, False),
        (Tool.BUILD_SIGNATURES, FULL | {"rows": 257}, False),
        (Tool.BUILD_SIGNATURES, FULL | {"rows": 64, "max_jaccard": ".005"}, False),
        (Tool.DESCRIBE_CORPUS, {"dataset": "acl-full", "min_papers": 50, "threshold": "0.70"}, True),
        (Tool.DESCRIBE_CORPUS, {"dataset": "acl-full", "min_papers": 0, "threshold": "0.70"}, False),
    ],
)
def test_checks_what_the_types_cannot_say(tool: Tool, value: dict, valid: bool) -> None:
    value = {key: item for key, item in value.items() if key in TOOLS[tool][0].__dataclass_fields__}
    assert verify_input(tool, value)["valid"] is valid


@pytest.mark.parametrize(
    ("changes", "valid"),
    [({}, True), ({"signature_id": "acl-full-r8"}, False), ({"signature_id": "acl-small-8"}, False),
     ({"count": 0}, False), ({"threshold": "0"}, False)],
)
def test_checks_a_check_call(changes: dict, valid: bool) -> None:
    value = {"dataset": "acl-small", "signature_id": "acl-small-r64", "start_rank": 1, "count": 10}
    task = {key: SMALL[key] for key in ("min_papers", "threshold", "max_jaccard", "cost_budget")}
    assert verify_input(Tool.CHECK_PAIRS, value | task | changes)["valid"] is valid
