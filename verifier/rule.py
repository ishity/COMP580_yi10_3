"""The exact rules: decimals, similarity, nesting, gaps, costs, the histogram and the workspace file names, in integers
and Fractions only.

src/plugin/rule.py and verifier/rule.py hold this same file, so the tools and the verifier decide alike.
"""

import re
from fractions import Fraction

DECIMAL = re.compile(r"^[0-9]+(?:\.[0-9]{1,3})?$")
DECIMAL_RULE = 'a decimal string in (0, 1] with at most 3 decimal places, e.g. "0.80"'
HASH_ROWS = 256  # how many MinHash rows src/plugin/data/hashes.json holds
ROW_RATE = "0.001"  # r: one signature entry costs a thousandth of one exact posting read
EXACT_BUCKETS = 16  # at most this many distinct matching-row counts get one bucket each
NON_ALNUM = re.compile(r"[^a-z0-9]+")
LEDGERS = "ledgers"  # one ledger per set of locked task values: ledgers/<ledger_key>.json
CANDIDATES = "candidates"  # each ledger's rankings: candidates/<ledger_key>/<signature_id>.json


def parse_decimal(text):
    """``text`` as a Fraction, or None when it is not DECIMAL_RULE."""
    if not DECIMAL.fullmatch(text):
        return None
    value = Fraction(text)
    return value if 0 < value <= 1 else None


def three_places(value):
    """A DECIMAL_RULE value (a Fraction) written by value to 3 places, e.g. Fraction(7, 10) -> "0.700"."""
    scaled = value * 1000
    if scaled.denominator != 1:
        raise ValueError(f"{value} has more than 3 decimal places")
    return f"{scaled.numerator // 1000}.{scaled.numerator % 1000:03d}"


def ledger_key(dataset, min_papers, threshold, max_jaccard, cost_budget):
    """The name of the ledger for these locked task values (``dataset`` a string, the two decimals Fractions), e.g.
    "acl-full-m50-t0.700-j0.005-b24000": different task values never share a ledger, and equal values always do."""
    return f"{dataset}-m{min_papers}-t{three_places(threshold)}-j{three_places(max_jaccard)}-b{cost_budget}"


def ledger_path(key):
    """The ledger's path in the workspace."""
    return f"{LEDGERS}/{key}.json"


def candidates_path(key, signature_id):
    """The path in the workspace of the ranking that signatures ``signature_id`` saved under ledger ``key``."""
    return f"{CANDIDATES}/{key}/{signature_id}.json"


def similar(dot, norm_a, norm_b, threshold):
    """Whether two embeddings' cosine, dot / sqrt(norm_a * norm_b), is at least ``threshold`` (a Fraction)."""
    return dot >= 0 and dot * dot >= threshold * threshold * norm_a * norm_b


def phrase(name):
    """A concept name's normalized tokens, space-padded: lowercase, every character outside a-z and 0-9 a space,
    and a token longer than 3 characters ending in "s", but not "ss", "us" or "is", without that "s"."""
    tokens = NON_ALNUM.sub(" ", name.lower()).split()
    stems = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")) else t for t in tokens]
    return " " + " ".join(stems) + " "


def nested(a, b):
    """Whether one concept's phrase occurs as whole tokens inside the other's, e.g. translation and machine translation."""
    return phrase(a) in phrase(b) or phrase(b) in phrase(a)


def is_gap(intersection, union, max_jaccard):
    """Whether a related pair's Jaccard, intersection / union, is at most ``max_jaccard`` (a Fraction)."""
    return union > 0 and Fraction(intersection, union) <= max_jaccard


def signature_cost(rows, postings):
    """What signatures of ``rows`` rows cost: rows times the frequent concepts' postings times ROW_RATE, rounded up."""
    rate = Fraction(ROW_RATE)
    return -(-rows * postings * rate.numerator // rate.denominator)


def signature_charge(rows, built, postings):
    """What a build of ``rows`` rows is charged when the longest signatures built so far have ``built`` rows: only the
    rows beyond ``built``, since the first rows of longer signatures are the shorter signatures."""
    return signature_cost(rows, postings) - signature_cost(built, postings) if rows > built else 0


def estimate(matches, rows):
    """matches / rows as a decimal string truncated to 4 places, e.g. 1 / 3 -> "0.3333"."""
    scaled = matches * 10000 // rows
    return f"{scaled // 10000}.{scaled % 10000:04d}"


def histogram(ranked, rows):
    """Buckets of ranked (matches, check_cost) pairs, fewest matches first, as (min_matches, max_matches,
    min_estimate, max_estimate, pairs, check_cost, first_rank, last_rank): one per distinct matches value when there
    are at most EXACT_BUCKETS of them, else one per range 0, 1, 2-3, 4-7, 8-15 and so on."""
    if len({matches for matches, _ in ranked}) <= EXACT_BUCKETS:
        keys = [matches for matches, _ in ranked]
    else:
        keys = [matches.bit_length() for matches, _ in ranked]
    buckets, start = [], 0
    for rank in range(1, len(ranked) + 1):
        if rank == len(ranked) or keys[rank] != keys[start]:
            group = ranked[start:rank]
            low, high = group[0][0], group[-1][0]
            cost = sum(check_cost for _, check_cost in group)
            buckets.append((low, high, estimate(low, rows), estimate(high, rows), len(group), cost, start + 1, rank))
            start = rank
    return buckets
