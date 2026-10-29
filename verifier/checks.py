"""Checks shared by the tool verifiers. Pure functions of their arguments and the files on disk."""

import json
import re
from fractions import Fraction
from pathlib import Path

from .rule import DECIMAL_RULE, histogram, parse_decimal, signature_charge

LEDGER = "ledger.json"
CANDIDATES = "candidates"


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def decimals_problem(**texts):
    for name, text in texts.items():
        if parse_decimal(text) is None:
            return f"{name} must be {DECIMAL_RULE}, got {text!r}"
    return None


def signature_id_problem(dataset, signature_id):
    if not re.fullmatch(re.escape(dataset.value) + r"-r[1-9][0-9]*", signature_id):
        return f"signature_id {signature_id!r} is not a {dataset.value} signature, {dataset.value}-r<rows>"
    return None


def read_json(path):
    """The JSON in ``path``, or None when there is no such file."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def locked_problem(ledger, value):
    """What differs between the task values the ledger fixed and those ``value`` passes, or None."""
    if (ledger["dataset"], ledger["min_papers"], ledger["cost_budget"]) != (
        value.dataset.value, value.min_papers, value.cost_budget
    ):
        return "the ledger is for another dataset, min_papers or cost_budget"
    for name, text in (("threshold", value.threshold), ("max_jaccard", value.max_jaccard)):
        given = parse_decimal(text)
        if given is None or parse_decimal(ledger[name]) != given:
            return f"the ledger is for another {name}"
    return None


def charges_problem(ledger):
    """What is wrong with the ledger's signature charges: each build pays only for rows beyond the longest so far."""
    built = 0
    for call in ledger["signatures"]:
        if call["signature_id"] != f"{ledger['dataset']}-r{call['rows']}":
            return f"the ledger names signatures {call['signature_id']} for another rows"
        expected = signature_charge(call["rows"], built, ledger["frequent_postings"])
        if call["charged"] != expected:
            return f"the ledger charges signatures {call['signature_id']} {call['charged']}, not {expected}"
        built = max(built, call["rows"])
    return None


def cost_used(ledger):
    return sum(call["charged"] for call in ledger["signatures"]) + sum(call["charged"] for call in ledger["checks"])


def candidates_problem(saved, signature_id, value):
    """What is wrong with saved signatures' ranked related pairs, or None."""
    if (saved["signature_id"], saved["dataset"], saved["rows"], saved["min_papers"]) != (
        signature_id, value.dataset.value, value.rows, value.min_papers
    ) or parse_decimal(saved["threshold"]) != parse_decimal(value.threshold):
        return "the saved pairs are not for the requested signatures, min_papers and threshold"
    names = [(a, b) for _, a, b, _ in saved["pairs"]]
    if len(set(names)) != len(names):
        return "the saved signatures list a pair twice"
    for matches, a, b, cost in saved["pairs"]:
        if not a < b:
            return f"pair ({a!r}, {b!r}) is not in canonical order, a before b"
        if not 0 <= matches <= value.rows:
            return f"pair ({a!r}, {b!r}) matches {matches} rows, not from 0 to {value.rows}"
        if cost < 2 * value.min_papers:
            return f"pair ({a!r}, {b!r}) costs {cost}, below 2 * min_papers, though both concepts are frequent"
    keys = [(matches, a, b) for matches, a, b, _ in saved["pairs"]]
    if keys != sorted(keys):
        return "the saved pairs are not ranked by matching rows, fewest first, then by (a, b)"
    return None


def histogram_of(saved):
    return histogram([(matches, cost) for matches, _, _, cost in saved["pairs"]], saved["rows"])


def gap_problem(pair, max_jaccard):
    """What is wrong with a pair returned as a gap, judged by its own integers, or None."""
    if not 0 <= pair["intersection"] <= pair["union"] or pair["union"] < 1:
        return f"pair ({pair['a']!r}, {pair['b']!r}) has impossible sizes {pair['intersection']} and {pair['union']}"
    if Fraction(pair["intersection"], pair["union"]) > max_jaccard:
        return f"pair ({pair['a']!r}, {pair['b']!r}) has Jaccard {pair['intersection']}/{pair['union']}, above max_jaccard"
    return None
