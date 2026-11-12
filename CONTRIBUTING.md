# Contributing

## Work from this repository

Clone the new `main` branch. Do not merge or push branches from earlier private
development repositories: they have unrelated histories. Transfer pending
changes as reviewed file patches on a new branch instead.

```bash
git clone https://github.com/adminblackjack/OutfitterBench.git
cd OutfitterBench
uv sync --extra dev --frozen
git switch -c fix/short-description
uv run pytest
```

Keep changes focused. Explain the behavior affected and add regression tests.
Before pushing, inspect `git diff --cached` and verify that no credentials,
local configurations, run databases, model transcripts, or internal documents
are staged. Stage specific files rather than an entire research workspace.

```bash
git add path/to/changed_file path/to/test_file
git diff --cached
git commit -m "Describe the change"
git push -u origin fix/short-description
```

## Preserve benchmark identity

The published experiment uses the V54 rental configuration. Changing demand,
customer responses, prices, resource dynamics, scoring, or actor-visible rules
requires an explicit environment-version decision. Packaging and analysis fixes
must not silently alter the world.

This distribution uses the separately versioned public weather described in
[docs/WEATHER.md](docs/WEATHER.md). Retain the weather dataset identity when
reporting scores. Do not label new public-weather scores as reproductions of the
paper's original-weather scores, even when the same seed is used.

For weather changes, preserve the source snapshots and attribution, update the
dataset identity, and run the builder checks and relevant regression tests.
Source observations are never adjusted merely to match a model's score.

Keep source and runtime-data integrity checks intact. Do not replace recorded
hashes to make an incompatible trajectory resume or replay. Retain the matching
code and frozen data for historical runs. The optional alternate assortment
configuration is not part of the paper's reported results.

## Tests and data

Run `uv run pytest` and the relevant no-LLM installation check from the README.
Do not add a paid API call to the default test suite. Expensive evaluation stays
opt-in. Integration tests requiring private trajectories must not run by default.

Runtime data must be included in built packages, with provenance and integrity
metadata preserved. The code license does not grant rights to third-party data;
review [docs/DATA.md](docs/DATA.md) before adding or distributing new inputs.

## Public release

The repository remains private until the maintainer explicitly changes its
visibility. Before that change, complete [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).
Never upload local Git backups or old branches to this repository.
Old clones must point to the private archive, not this repository's reused URL.
Use a fresh clone for public development; transfer changes as reviewed patches,
not merges of the old history.
