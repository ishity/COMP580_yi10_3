# CLAUDE.md: build the `concept-gap-detection` NanoAgent plugin

## Context

This is plugin 3 of 5 for the COMP 480/580 milestone "Verifiable Tasks" (due 2026-11-02). Plugins 1 and 2 are finished, reviewed and live on RUSH Lab:

- Plugin 1, `lsh-similar-pairs` (local: `~/lsh-similar-pairs`): SimHash LSH over 493 keyword embeddings, exact integer cosine rule.
- Plugin 2, `concept-cooccurrence-query` (local: `~/COMP580_yi10_2`): Count-Min Sketch over the shared ACL Anthology corpus.

Reuse their structure, patterns and helpers wherever they fit.

Goal: an original task, **finding research gaps: concept pairs that are individually frequent and semantically similar but rarely studied together**, measured by the Jaccard similarity of their paper sets and estimated with **MinHash**, under a single cost budget that the agent splits between signature length and exact checks. A deterministic verifier checks the result against an exact reference.

This is the core definition of GapMap (a gap is a combination of concepts that are each well studied but rarely explored together), made checkable with numeric thresholds.

## Read first

1. `~/nanoagent/docs/instructions/skills/publish-plugin/SKILL.md` (authoritative rules).
2. `~/COMP580_yi10_2` in full (`CLAUDE.md`, `contract.py`, `nanoagent.yaml`, `src/plugin/`, `verifier/`, `tests/`) and `~/cooccurrence-report-notes.md`.
3. `~/lsh-similar-pairs/src/plugin/rule.py`, its data format, and `~/lsh-report-notes.md`.

All design decisions recorded there apply here too: ledger locking of task values, dedup of repeated items (not charged twice), rank ranges instead of item lists, candidates in workspace files, informative errors that state the valid value, decimal strings parsed with Fraction, the reference tool taking the full task input, the verifier rederiving every call from the workspace files.

If anything here conflicts with SKILL.md, SKILL.md wins. Tell me about the conflict.

## Hard rules

- **uv only.** No pip, conda or pixi. Keep `requires-python`, dependencies and the `slimconfig` source unchanged except for the package-data setting.
- **Verifier**: pure standard library plus `contract`; no LLM, clock, randomness, environment, network, subprocess, `open`, `eval`, `exec`, `getattr`, private attributes or writes. `Path(...).read_text()` only. Deterministic across two runs.
- **Rules in code, judgment to the model. No fallbacks. Only allowed top-level files.**
- **Item lists never pass through the model** except the final submission, which must stay at about 150 pairs or fewer per instance.
- **Never run `git push` or `gh repo create`, and never make anything public. I commit and push myself.** You may make local commits only when I say so.
- No em-dashes in any text you write.

## Setup

```bash
cd ~
rsync -a --exclude .git --exclude .venv --exclude CLAUDE.md COMP580_yi10_2/ COMP580_yi10_3/
cd COMP580_yi10_3
git init -b main
```

This file becomes the repo's `CLAUDE.md`. Rename everywhere (`nanoagent.yaml` id `concept-gap-detection` and name "Concept Gap Detection", README title, `pyproject.toml` name `nanoagent-concept-gap-detection`), version 0.1.0, `status: in-development`, LICENSE stays "Yuri", 2026. Remove every Count-Min-specific tool, task, test, verifier module and data file.

## Data (derived outside the repo)

Add a script `~/gapmap-corpus-gen/derive_gaps.py` to the existing corpus project. It reads the master corpus and the plugin 1 embeddings, and writes this plugin's JSON into `src/plugin/data/`.

- **Vocabulary alignment**: confirm that the 493 keywords of the corpus and of plugin 1's `nlp-large` embeddings are the same strings in the same order. If not, map by string and report any mismatch at Checkpoint 1.
- **Per dataset**, bundle: for each concept, its name, its integer embedding (from plugin 1, unchanged), and its sorted list of paper indices. Paper indices are 0-based positions in the dataset's paper list.
- **Datasets** (enum `Dataset`): `acl-small` (the same 2018 and 2019 papers as plugin 2), `acl-full` (all papers), and `edge-cases` (hand-built, see Tests).
- `hashes.json`: fixed parameters for up to 256 MinHash rows, generated once with a seed: `h_i(x) = (a_i * x + b_i) mod p` with p = 2^31 - 1, x the paper index. Record the seed.

## Exact rules (shared by tools and verifier, word for word in `src/plugin/rule.py` and `verifier/rule.py`)

- **Frequent**: a concept is frequent when it appears in at least `min_papers` papers.
- **Similar**: plugin 1's integer cosine rule: `dot >= 0` and `dot^2 >= tau^2 * nu * nv`, compared as Fractions.
- **Nested pairs** (one concept's normalized phrase inside the other's, as defined in plugin 2) are never related pairs.
- **Related pair**: two distinct frequent concepts that are similar and not nested.
- **Gap**: a related pair whose Jaccard similarity `|A ∩ B| / |A ∪ B|` of paper sets is at most `max_jaccard`, compared as Fractions.

The verifier cannot read the bundled data, so the reference output must carry what it needs (as in plugins 1 and 2): the concept names, the related pairs, and for every related pair the integers `|A ∩ B|` and `|A ∪ B|`.

## Task design

### The task in one sentence

Given a corpus, a frequency floor, a similarity threshold and a Jaccard ceiling, find every related concept pair whose paper sets overlap at most that much. MinHash signatures estimate each related pair's Jaccard similarity; exact checks confirm gaps; one cost budget pays for both.

### Why it is objectively verifiable

The answer is a set fully determined by the bundled data and the four thresholds, decided with integers and Fractions only. Precision, recall and the budget are checkable by code.

### Cost model (calibrate at Checkpoint 2)

One budget, `cost_budget`, in abstract units. Building MinHash signatures with `rows` hash rows costs `rows × F × r` units, where F is the number of frequent concepts and r a fixed per-signature-entry rate; an exact Jaccard check of one pair costs 1 unit. Propose r at Checkpoint 2 so that the agent faces a real trade-off: very short signatures rank poorly, very long ones leave too little for exact checks. Building the same signature length again is not charged twice. Every cost is recorded in the ledger.

### Tools

| Tool | Input | Output | Notes |
|---|---|---|---|
| `describe_corpus` | `dataset`, `min_papers`, `threshold` | paper count, concept count, frequent concept count F, related pair count, hash rows available, the per-entry rate r | Pure description; charges nothing. |
| `build_signatures` | `dataset`, `rows`, `min_papers`, `threshold`, `max_jaccard`, `cost_budget` | `signature_id`, cost charged and remaining, related pair count, histogram of related pairs by estimated Jaccard | Computes a MinHash signature of `rows` rows for every frequent concept, estimates each related pair's Jaccard as the share of matching rows, and saves all related pairs to `workspace/candidates/<signature_id>.json`, ranked by estimate ascending (most likely gaps first), ties by (a, b) name order. The ledger locks dataset, min_papers, threshold, max_jaccard and cost_budget on the first call. Histogram: one bucket per number of matching rows if at most 16 distinct values, else propose an integer-only bucketing; each bucket with lowest and highest estimate (as decimal strings), count, first_rank, last_rank. |
| `check_pairs` | `dataset`, `signature_id`, `start_rank`, `count`, plus the locked task values | gap pairs with their exact intersection and union sizes, checked, skipped, total checked, cost remaining | Exact Jaccard for a rank range. Pairs checked by any earlier call are skipped and not charged. Refuses a range whose new pairs exceed the remaining budget, saying how many new pairs it holds and the largest count that fits. |
| `exact_gaps` | full task input | concept names, all related pairs with their intersection and union sizes, and the gap pairs | **Reference only.** Not in the task's `tools`. |

`signature_id` is deterministic, for example `<dataset>-r<rows>`.

### Task

- id `find-concept-gaps`, runner `agent`, tools `[describe_corpus, build_signatures, check_pairs]`, reference `exact_gaps`.
- Input: `dataset`, `min_papers` (integer), `threshold` (decimal string, cosine), `max_jaccard` (decimal string), `cost_budget` (integer), `min_recall` (decimal string).
- Output: the list of gap pairs (concept names, a < b).
- Instructions must explain: what frequent, similar, related and gap mean; what MinHash estimates (the probability that one row matches equals the Jaccard similarity, so the estimate's standard deviation is about sqrt(J(1 - J) / rows)); that pairs which never co-occur have Jaccard 0 and their rows never match; how the single budget is spent and that it is shared; what the histogram shows; that a ranking may hold more pairs than the budget can check; and that an empty list is correct when no pair qualifies. Never name a configuration that works.

### Verifier

Same order as plugins 1 and 2, adapted:
1. Malformed: known concepts, a ≠ b, no duplicates in either order.
2. Budget: ledger matches every locked task value; total cost within `cost_budget`; non-empty answer without a ledger fails.
3. Precision: every submitted pair is a gap in the reference.
4. Provenance: `check_pairs` returned every submitted pair as a gap, with the reference's intersection and union sizes.
5. Recall: as before; score = recall.

`verify_output` rederives each call from the saved candidate files and the ledger: ranking order, histogram, costs, skipped and checked counts, and that every returned gap satisfies the Jaccard rule by its own integers.

### Tests

Mirror plugin 2's coverage, plus: a pair with Jaccard exactly `max_jaccard`; a pair just above it; a pair that never co-occurs (Jaccard 0) whose rows never match; a concept at exactly `min_papers` and one just below; a similar but nested pair excluded from related pairs; cost exceeded by signatures alone; cost exceeded by checks; rebuilding the same signature not charged twice; the reference equals a slow independent computation on `acl-small`.

`edge-cases` must contain each of these situations by construction.

## Work plan and checkpoints

Stop at each checkpoint, summarize, and wait for my OK.

1. Setup and rename. Vocabulary alignment check. Derive datasets. **Checkpoint 1**: for `min_papers` in a few values (for example 50, 100, 200) and `threshold` in {0.70, 0.75, 0.80}: F, related pairs, and the distribution of their exact Jaccard (how many at exactly 0, and quantiles), plus the 15 related pairs with the lowest Jaccard as a sanity check that they look like plausible research gaps.
2. Choose candidate thresholds so that each instance has at most about 150 gaps and the gaps are a clear minority of related pairs. Simulate recall for several `rows` values and budget splits, and propose r and `cost_budget`. **Checkpoint 2**: tables and proposed instances (two main instances, one with an empty reference, and edge-cases), each showing which settings pass and which fail.
3. `contract.py`. **Checkpoint 3**: show it with any design decision you had to make.
4. Tools, manifest, verifier, tests.
5. UI, product page, README (`## Tools`, `## Layout`, `## Develop and run`, the instance table, the exact rules, the verifier order, the data source and license).
6. All checks: `uv lock`, `uv sync --extra dev`, `uv run pytest`, `uv run ruff check .`, `uv lock --check`, `nanoagent plugin inspect .`, SKILL.md section 8. Status `ready`. Confirm the tools reproduce the Checkpoint 2 numbers, and run a scripted agent end to end through the real MCP server and sandboxed verifier. **Checkpoint 4**: results, plus the exact instruction and tool text the model will see.

Do not commit or push unless I say so.

## Report notes

Keep them in `~/gap-report-notes.md`, never in this repo: data derivation command, vocabulary alignment result, chosen thresholds and cost model with reasons, every verifier case and verdict, test output, and limitations (for example: Jaccard of paper sets is one of several reasonable rarity measures; concepts come from exact phrase matching; semantic similarity comes from one embedding model projected to 32 dimensions; the cost model is abstract, since the tool itself could compute every Jaccard exactly).