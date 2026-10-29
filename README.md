# Concept Gap Detection

A NanoAgent plugin with one checkable job: find research gaps in a bundled corpus of ACL Anthology papers.
A gap is a pair of concepts that are each well studied and close in meaning, but rarely studied together:
both concepts are frequent, their embeddings are similar, and the Jaccard similarity of their paper sets is
at most a ceiling. MinHash signatures estimate each related pair's Jaccard, exact checks confirm the gaps,
and one cost budget pays for both. The plugin is a standard MCP server over stdio; NanoAgent owns its
launch command, environment, input and output checks and lifecycle.

## Tools

| Tool | Does | Code |
|---|---|---|
| `describe_corpus` | Gives a corpus's paper and concept counts, the frequent concepts F and their postings P, the related pairs and what checking them all would cost, the MinHash rows (256) and the signature rate r (0.001); charges nothing | `src/plugin/describe.py` |
| `build_signatures` | Builds `rows`-row MinHash signatures of the frequent concepts, charges only the rows beyond the longest built so far, ranks every related pair by matching rows (fewest first) into `candidates/<signature_id>.json` in the workspace, and returns the id, the cost charged, used and left, and a histogram of the ranking | `src/plugin/signatures.py` |
| `check_pairs` | Measures a rank range of a saved ranking exactly, charges `\|A\| + \|B\|` per pair not checked before, and returns the gaps with their intersection and union sizes; records every call in `ledger.json` and refuses a range over the budget | `src/plugin/check.py` |
| `exact_gaps` | Measures every related pair exactly and lists the gaps: the task's reference, which the model never sees | `src/plugin/exact.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `find-concept-gaps` | Finds every gap of a corpus within one cost budget for signatures and exact checks | `agent`, with `describe_corpus`, `build_signatures` and `check_pairs`; reference `exact_gaps` |

The task input is a `dataset`, `min_papers`, `threshold` (a cosine), `max_jaccard`, `cost_budget` and
`min_recall`. Decimals are strings in (0, 1] with at most 3 decimal places, such as `"0.70"`, compared as
fractions. The output is the list of gaps, each as two concept names.

### The exact rules

`src/plugin/rule.py` and `verifier/rule.py` hold the same file, word for word, in integers and fractions:

- **Frequent**: a concept is frequent when at least `min_papers` papers mention it.
- **Similar**: two integer embeddings with dot product `d` and squared lengths `n_u`, `n_v` are similar when
  `d ≥ 0` and `d² ≥ threshold² · n_u · n_v` (plugin 1's cosine rule).
- **Nested**: one concept's normalized phrase occurs as whole tokens in the other's (`translation`,
  `machine translation`). Normalizing lowercases, maps every character outside `a-z0-9` to a space, and drops
  a final `s` from tokens longer than 3 characters not ending in `ss`, `us` or `is` (plugin 2's rule).
- **Related pair**: two distinct frequent concepts that are similar and not nested.
- **Gap**: a related pair whose Jaccard `|A ∩ B| / |A ∪ B|` of paper sets is at most `max_jaccard`.

### Costs

- An exact check of a pair costs `|A| + |B|`, the posting reads of merging its two sorted paper lists.
- MinHash row `i` maps paper index `x` to `(a_i · x + b_i) mod p`, `p = 2³¹ − 1`, and a signature keeps
  each row's least value over the concept's papers. Building signatures of `rows` rows costs
  `⌈rows · P · r⌉`, with P the frequent concepts' postings and `r = 1/1000`. The first rows of longer
  signatures are the shorter signatures, so a build is charged
  `⌈rows · P · r⌉ − ⌈built · P · r⌉` when `rows` exceeds the longest `built` so far, and 0 otherwise.
- At `r = 1` one row costs about as much as checking every related pair, so MinHash never pays off at
  this corpus's size. `r = 1/1000` models signatures that are built once and reused across many queries.

A related pair's estimate is its share of matching rows, whose expectation is its Jaccard; pairs that never
share a paper never match, since each row is one-to-one on paper indices. Rankings never pass through the
model: the agent sees each ranking's histogram, with one bucket per matching-row count when there are at
most 16, else one per range 0, 1, 2–3, 4–7 and so on; each bucket gives its matching rows, its estimates
truncated to 4 places, its pair count, the cost of checking all of it and its rank range. The first
`build_signatures` call fixes the ledger's dataset, `min_papers`, `threshold`, `max_jaccard` and
`cost_budget`, and `check_pairs` must pass the same. A pair any earlier call checked is skipped and not
charged again. Every refused call says what was wrong and which value would be valid, for example the most
rows or the largest `count` that still fits.

### Verification

`verify_task` checks, in order, and the first failure decides:

1. Malformed: every pair names two distinct concepts of the dataset, and no pair is submitted twice,
   counting `(a, b)` and `(b, a)` as one.
2. Budget: the ledger is for the task's dataset, `min_papers`, `threshold`, `max_jaccard` and `cost_budget`
   and the reference's P; each signature build is charged as the incremental rule says; every saved ranking
   holds exactly the reference's related pairs at cost `intersection + union`; each check is charged the
   `|A| + |B|` of its pairs, no pair is checked twice, and the total stays within `cost_budget`. A non-empty
   answer without a ledger fails.
3. Precision: every submitted pair is a gap in the reference. Any other pair scores 0.
4. Provenance: `check_pairs` returned every submitted pair as a gap, with the reference's intersection and
   union sizes.
5. Recall: `recall = |submitted| / |gaps|` (1 for an empty answer to an empty reference). The result is
   valid when recall is at least `min_recall`, and its score is the recall.

### Instances

| Dataset | Papers | min_papers | threshold | max_jaccard | Related | Gaps | Check all | cost_budget | min_recall |
|---|---|---|---|---|---|---|---|---|---|
| `acl-full` | 21,934 | 50 | 0.70 | 0.005 | 69 | 23 | 53,234 | 24,000 | 0.90 |
| `acl-small` | 2,924 | 10 | 0.70 | 0.01 | 42 | 17 | 6,750 | 3,800 | 0.90 |
| `acl-small` | 2,924 | 50 | 0.70 | 0.005 | 7 | 0 | 2,027 | 1,000 | 1.0 |
| `edge-cases` | 76 | 4 | 0.80 | 0.25 | 6 | 3 | 65 | 60 | 1.0 |

With one signature and its longest affordable prefix checked, 64 or 128 rows pass both main instances,
while 32 rows or fewer, and 256 rows, fail. The empty instance's lowest related Jaccard is 0.0053, just
above its ceiling. `edge-cases` holds, each on its own pair of axes: a pair with Jaccard exactly 0.25 and
cosine exactly 0.80, one at 3/11 just above, a pair that never co-occurs, a concept at exactly 4 papers and
one at 3, a similar nested pair, a dissimilar pair, and pairs with Jaccard 1 and 0.5.

### Data

`acl-full` and `acl-small` come from ACL Anthology papers of 2016 to 2024 (2018 and 2019 for `acl-small`)
with an abstract and at least 2 of 493 NLP keywords in title and abstract, matched as whole phrases after
the normalization above. Each concept carries its 32-dimensional integer embedding (all-MiniLM-L6-v2,
randomly projected, from plugin 1) and its sorted paper indices. The generator lives outside this
repository; only its JSON output is here. `hashes.json` holds 256 rows `(a, b)` drawn with
`random.Random(580)`. Source: ACL Anthology, https://aclanthology.org/, CC BY 4.0; the data holds anthology
ids and concept paper indices only, no titles or abstracts.

## Layout

It has the standard plugin layout, which NanoAgent requires:

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # needed environment variable names (none here)
├── src/plugin/      # the MCP server (__main__.py, its IMPLEMENTATIONS table), one module per tool,
│                    # bundle.py (corpora, signatures, related pairs, ledger and paths), rule.py (the exact
│                    # rules) and data/ (the JSON)
├── tests/           # test_tools.py and test_verifier.py
├── ui/              # renderers.js and renderers.css: outputs are drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package, one module per tool, find.py for the task
```

`contract.py` types every value: the server builds each call's input into its class and checks its
output against its own, and NanoAgent and the verifier get the same classes. A failing tool raises,
and the MCP server returns an error result.

`verifier/` is deterministic: it imports only the standard library and `contract`, and reads only the
workspace's `candidates/` and `ledger.json`. Beyond the task check, it rederives every tool call: for
`build_signatures`, the signature id, each build's incremental charge, the cost used and left, and the
ranking order and histogram against the saved ranking; for each `check_pairs` call, that the ledger checked
exactly the range's new pairs at their saved cost, that every returned gap satisfies the Jaccard rule by its
own integers and that `intersection + union` equals its cost, and that the totals add up; for `exact_gaps`,
that no related pair is nested and that the gaps are exactly the related pairs within the ceiling.

## Develop and run

```bash
uv lock
uv sync --extra dev
uv run pytest
uv run ruff check .
uv lock --check
```

NanoAgent passes the folder the plugin may write to as `NANOAGENT_WORKSPACE`; `build_signatures` writes
`candidates/<signature_id>.json` there, and both `build_signatures` and `check_pairs` write `ledger.json`.
