"""The bundled corpora and MinHash rows, the related pairs, and the names and paths the tools share."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

from contract import Dataset
from plugin import rule
from plugin.rule import DECIMAL_RULE, nested, parse_decimal, similar

DATA = Path(__file__).with_name("data")
LOCKED = ("dataset", "min_papers", "threshold", "max_jaccard", "cost_budget")


@dataclass(frozen=True)
class Corpus:
    names: tuple[str, ...]
    embeddings: tuple[tuple[int, ...], ...]
    norms: tuple[int, ...]
    # Each concept's papers: sorted 0-based indices into the dataset's paper list.
    papers: tuple[tuple[int, ...], ...]
    paper_count: int


@dataclass(frozen=True)
class Related:
    """A related pair of concept indices, its names in canonical order, and its exact paper set sizes."""

    i: int
    j: int
    a: str
    b: str
    intersection: int
    union: int

    @property
    def check_cost(self) -> int:
        """|A| + |B|, the posting reads of merging the two sorted paper lists."""
        return self.intersection + self.union


@cache
def load(dataset: Dataset) -> Corpus:
    raw = json.loads((DATA / f"{dataset.value}.json").read_text())
    concepts = raw["concepts"]
    names = tuple(concept["name"] for concept in concepts)
    if len(set(names)) != len(names):
        raise ValueError(f"{dataset.value} must hold unique concepts")
    embeddings = tuple(tuple(concept["embedding"]) for concept in concepts)
    norms = tuple(sum(x * x for x in vector) for vector in embeddings)
    if not all(norms):
        raise ValueError(f"{dataset.value} holds a zero embedding, whose cosine is undefined")
    papers = tuple(tuple(concept["papers"]) for concept in concepts)
    count = len(raw["paper_ids"])
    for name, held in zip(names, papers):
        if list(held) != sorted(set(held)) or (held and not 0 <= held[0] <= held[-1] < count):
            raise ValueError(f"{dataset.value} concept {name!r} must list distinct sorted paper indices in range")
    return Corpus(names=names, embeddings=embeddings, norms=norms, papers=papers, paper_count=count)


@cache
def hash_rows() -> tuple[int, tuple[tuple[int, int], ...]]:
    """The prime p and each row's (a, b), for h(x) = (a * x + b) mod p."""
    raw = json.loads((DATA / "hashes.json").read_text())
    return raw["prime"], tuple((row["a"], row["b"]) for row in raw["rows"])


@cache
def signature(dataset: Dataset, concept: int) -> tuple[int, ...]:
    """A concept's full MinHash signature: per row, the least h(x) over its papers x. Shorter signatures are its
    first rows."""
    prime, rows = hash_rows()
    held = load(dataset).papers[concept]
    return tuple(min((a * x + b) % prime for x in held) for a, b in rows)


def frequent(corpus: Corpus, min_papers: int) -> list[int]:
    return [i for i, held in enumerate(corpus.papers) if len(held) >= min_papers]


@cache
def related_pairs(dataset: Dataset, min_papers: int, threshold: Fraction) -> tuple[Related, ...]:
    """Every pair of distinct frequent concepts that are similar and not nested, by concept index."""
    corpus = load(dataset)
    chosen = frequent(corpus, min_papers)
    found = []
    for x, i in enumerate(chosen):
        for j in chosen[x + 1 :]:
            dot = sum(u * v for u, v in zip(corpus.embeddings[i], corpus.embeddings[j]))
            if not similar(dot, corpus.norms[i], corpus.norms[j], threshold):
                continue
            if nested(corpus.names[i], corpus.names[j]):
                continue
            a, b = sorted((corpus.names[i], corpus.names[j]))
            inter = len(set(corpus.papers[i]) & set(corpus.papers[j]))
            union = len(corpus.papers[i]) + len(corpus.papers[j]) - inter
            found.append(Related(i=i, j=j, a=a, b=b, intersection=inter, union=union))
    return tuple(found)


def postings(corpus: Corpus, min_papers: int) -> int:
    """P: how many (concept, paper) entries the frequent concepts hold in all."""
    return sum(len(corpus.papers[i]) for i in frequent(corpus, min_papers))


def decimal(text: str, name: str) -> Fraction:
    value = parse_decimal(text)
    if value is None:
        raise ValueError(f"{name} must be {DECIMAL_RULE}, got {text!r}")
    return value


def positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}: copy it from the task input")


def signature_id(dataset: Dataset, rows: int) -> str:
    return f"{dataset.value}-r{rows}"


def signature_id_pattern(dataset: Dataset) -> str:
    return re.escape(dataset.value) + r"-r[1-9][0-9]*"


def ledger_key(given) -> str:
    """The name of the ledger for the task values ``given`` passes: each instance keeps its own ledger, so a ledger an
    earlier run of another instance left in the workspace never locks this one."""
    return rule.ledger_key(
        given.dataset.value,
        given.min_papers,
        decimal(given.threshold, "threshold"),
        decimal(given.max_jaccard, "max_jaccard"),
        given.cost_budget,
    )


def candidates_path(workspace: Path, key: str, signatures: str) -> Path:
    return workspace / rule.candidates_path(key, signatures)


def read_ledger(workspace: Path, key: str) -> dict | None:
    path = workspace / rule.ledger_path(key)
    return json.loads(path.read_text()) if path.exists() else None


def write_ledger(workspace: Path, key: str, ledger: dict) -> None:
    path = workspace / rule.ledger_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger) + "\n")


def check_locked(ledger: dict, given) -> None:
    """Refuse a call whose task values differ from those the ledger records; decimals compare by value. The ledger's
    name already holds these values, so this only catches a ledger whose contents disagree with its name."""
    for field in LOCKED:
        value = getattr(given, field)
        fixed = ledger[field]
        if field == "dataset":
            same = fixed == value.value
        elif field in ("threshold", "max_jaccard"):
            same = Fraction(fixed) == decimal(value, field)
        else:
            same = fixed == value
        if not same:
            shown = f'"{fixed}"' if field in ("threshold", "max_jaccard") else fixed
            raise ValueError(f"this task's ledger is for {field} {shown}: pass {field} {shown}, copied from the task input")


def cost_used(ledger: dict) -> int:
    return sum(call["charged"] for call in ledger["signatures"]) + sum(call["charged"] for call in ledger["checks"])
