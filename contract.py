"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    DESCRIBE_CORPUS = "describe_corpus"
    BUILD_SIGNATURES = "build_signatures"
    CHECK_PAIRS = "check_pairs"
    EXACT_GAPS = "exact_gaps"


class Task(Enum):
    FIND_CONCEPT_GAPS = "find-concept-gaps"


class Dataset(Enum):
    ACL_SMALL = "acl-small"
    ACL_FULL = "acl-full"
    EDGE_CASES = "edge-cases"


DATASET = "The bundled corpus: its concepts, each with an embedding and the papers that mention it"
MIN_PAPERS = "The task input's min_papers, copied as given: a concept is frequent when at least this many papers mention it"
THRESHOLD = "The task input's threshold, copied as given: two concepts are similar when their cosine is at least this"
MAX_JACCARD = "The task input's max_jaccard, copied as given: a related pair is a gap when its Jaccard is at most this"
COST_BUDGET = "The task input's cost_budget, copied as given: the most cost signatures and checks may spend together"
SIGNATURE_ID = "The signatures' id, <dataset>-r<rows>, as build_signatures returned it"
CONCEPT_A = "The pair's first concept, the one that sorts first"
CONCEPT_B = "The pair's second concept, the one that sorts second"
INTERSECTION = "How many papers mention both concepts, |A ∩ B|, counted exactly"
UNION = "How many papers mention either concept, |A ∪ B|, counted exactly"


@dataclass
class Pair(Output):
    a: Annotated[str, CONCEPT_A]
    b: Annotated[str, CONCEPT_B]


@dataclass
class MeasuredPair(Output):
    a: Annotated[str, CONCEPT_A]
    b: Annotated[str, CONCEPT_B]
    intersection: Annotated[int, INTERSECTION]
    union: Annotated[int, UNION]


@dataclass
class FoundGap(Output):
    a: Annotated[str, CONCEPT_A]
    b: Annotated[str, CONCEPT_B]
    intersection: Annotated[int, INTERSECTION]
    union: Annotated[int, UNION]
    previously_scored: Annotated[
        bool, "True when an earlier call, perhaps in an earlier run of this task, checked this pair: returned again from the ledger at no cost"
    ]


@dataclass
class MatchBucket(Output):
    min_matches: Annotated[int, "The fewest matching signature rows among this bucket's pairs"]
    max_matches: Annotated[int, "The most matching signature rows among this bucket's pairs"]
    min_estimate: Annotated[str, "min_matches divided by rows, the lowest estimated Jaccard, truncated to 4 places"]
    max_estimate: Annotated[str, "max_matches divided by rows, the highest estimated Jaccard, truncated to 4 places"]
    pairs: Annotated[int, "How many related pairs fall in this bucket"]
    check_cost: Annotated[int, "What checking every pair of this bucket exactly would cost: the sum of |A| + |B|"]
    first_rank: Annotated[int, "The rank of this bucket's first pair, from 1"]
    last_rank: Annotated[int, "The rank of this bucket's last pair"]


@dataclass
class DescribeInput(Output):
    dataset: Annotated[Dataset, DATASET]
    min_papers: Annotated[int, MIN_PAPERS]
    threshold: Annotated[str, THRESHOLD]


@dataclass
class DescribeOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    papers: Annotated[int, "How many papers the corpus holds"]
    concepts: Annotated[int, "How many concepts the corpus holds"]
    frequent_concepts: Annotated[int, "How many concepts at least min_papers papers mention, F"]
    frequent_postings: Annotated[int, "How many (concept, paper) entries the frequent concepts hold in all, P"]
    related_pairs: Annotated[int, "How many pairs of frequent concepts are similar and not nested"]
    related_check_cost: Annotated[int, "What checking every related pair exactly would cost: the sum of |A| + |B|"]
    hash_rows: Annotated[int, "How many fixed MinHash rows signatures may use, the largest rows"]
    row_rate: Annotated[str, "r, the cost of one signature entry, as a decimal string: rows signatures cost rows times P times r, rounded up"]


@dataclass
class SignaturesInput(Output):
    dataset: Annotated[Dataset, DATASET]
    rows: Annotated[int, "How many MinHash rows each signature has, from 1 to hash_rows"]
    min_papers: Annotated[int, MIN_PAPERS]
    threshold: Annotated[str, THRESHOLD]
    max_jaccard: Annotated[str, MAX_JACCARD]
    cost_budget: Annotated[int, COST_BUDGET]


@dataclass
class SignaturesOutput(Output):
    signature_id: Annotated[str, SIGNATURE_ID]
    cost_charged: Annotated[int, "What this call charged: rows times P times r rounded up, or 0 for signatures built before"]
    cost_used: Annotated[int, "What all signatures and checks so far cost together"]
    cost_remaining: Annotated[int, "How much more the cost budget allows"]
    related_pairs: Annotated[int, "How many related pairs the signatures rank"]
    histogram: Annotated[
        list[MatchBucket], "The related pairs grouped by matching rows, fewest first, in rank order"
    ]


@dataclass
class CheckInput(Output):
    dataset: Annotated[Dataset, DATASET]
    signature_id: Annotated[str, SIGNATURE_ID]
    start_rank: Annotated[int, "The rank of the first pair to check, from 1"]
    count: Annotated[int, "How many pairs to check from start_rank on, at least 1"]
    min_papers: Annotated[int, MIN_PAPERS]
    threshold: Annotated[str, THRESHOLD]
    max_jaccard: Annotated[str, MAX_JACCARD]
    cost_budget: Annotated[int, COST_BUDGET]


@dataclass
class CheckOutput(Output):
    gaps: Annotated[
        list[FoundGap], "The range's gaps in rank order: those checked now, and those an earlier call checked, marked previously_scored"
    ]
    checked: Annotated[int, "How many pairs in the range were checked now"]
    skipped: Annotated[int, "How many pairs in the range an earlier call checked, so not checked or charged again"]
    cost_charged: Annotated[int, "What this call charged: |A| + |B| summed over the pairs checked now"]
    total_checked: Annotated[int, "How many distinct pairs all calls so far have checked"]
    cost_used: Annotated[int, "What all signatures and checks so far cost together"]
    cost_remaining: Annotated[int, "How much more the cost budget allows"]


@dataclass
class FindInput(Output):
    dataset: Annotated[Dataset, DATASET]
    min_papers: Annotated[int, "The least number of papers that must mention a concept for it to be frequent"]
    threshold: Annotated[str, "The least cosine of two similar concepts, as a decimal string, e.g. 0.70"]
    max_jaccard: Annotated[str, "The most Jaccard similarity of a gap's paper sets, as a decimal string, e.g. 0.01"]
    cost_budget: Annotated[int, "The most cost signatures and exact checks may spend together"]
    min_recall: Annotated[str, "The least share of all gaps to find, as a decimal string, e.g. 0.90"]


@dataclass
class ExactOutput(Output):
    dataset: Annotated[Dataset, DATASET]
    min_papers: Annotated[int, MIN_PAPERS]
    threshold: Annotated[str, THRESHOLD]
    max_jaccard: Annotated[str, MAX_JACCARD]
    concepts: Annotated[list[str], "Every concept of the dataset, in its order"]
    frequent_postings: Annotated[int, "How many (concept, paper) entries the frequent concepts hold in all, P"]
    related: Annotated[list[MeasuredPair], "Every related pair, measured exactly, sorted by a then b"]
    gaps: Annotated[list[Pair], "Every related pair whose Jaccard is at most max_jaccard, sorted by a then b"]


@dataclass
class FindOutput(Output):
    pairs: Annotated[list[Pair], "Every gap check_pairs returned, each once"]


TOOLS = {
    Tool.DESCRIBE_CORPUS: (DescribeInput, DescribeOutput),
    Tool.BUILD_SIGNATURES: (SignaturesInput, SignaturesOutput),
    Tool.CHECK_PAIRS: (CheckInput, CheckOutput),
    Tool.EXACT_GAPS: (FindInput, ExactOutput),
}
TASKS = {Task.FIND_CONCEPT_GAPS: (FindInput, FindOutput)}
