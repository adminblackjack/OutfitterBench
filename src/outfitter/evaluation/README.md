# Offline trajectory evaluation

The official business score and actor environment are unchanged. Metrics describe
service delivery, resource coverage, commitment protection, available choices,
capital use and execution consistency; they are not combined into another reward.

## Usage

```bash
outfitter-bench evaluate RUN_DIR --through-day 100
outfitter-bench evaluate RUN_DIR --through-day 100 --extended --source-root FROZEN_SOURCE
outfitter-bench evaluate RUN_A RUN_B --extended --source-root FROZEN_SOURCE --workers 2
outfitter-bench evaluate RUN_DIR --through-days 100 214 --extended --source-root FROZEN_SOURCE --cpu-budget 24
```

`--extended` is **off by default**. `analyze` only attaches basic results and never
starts historical replay. No command here uses an LLM or a judge. Extended worker
processes prohibit socket connections. `--source-root` accepts a source root or its
`src` directory; mismatching environment identities fail instead of silently
replaying a different version. Different source identities require separate calls.

Outputs are under `RUN_DIR/analysis/evaluation/outfitter-trajectory-metrics-v1/day-N/`:
`metrics.json`, `metrics.csv`, `REPORT.md`, object-level CSVs and an optional figure.
Use `--output-dir` for a single run, `--no-plots` to omit PNGs. Basic rescans preserve
complete extended reports by writing to a `basic` child directory.

All metric timestamps refer to completed days. A requested cutoff beyond the last
completed day fails. Snapshot morning state is not substituted for closing cash or
business value. An unfinished season has no final evaluation score.

## Definitions

- Service reliability: pickups / (pickups + merchant cancellation + pickup failure
  + merchant-induced customer cancellation), unique contracts. External customer
  cancellation and pending commitments are separate. Legally reduced pickups count
  as delivered but are separated from full delivery without omission.
- Basic nominal calendar: exact physical units of all committed lines, including
  optional lines, over `[pickup, return + gap)` for gaps 0/1/2. Accepted owned orders
  count immediately; walk-ins, queues, damage and future life exhaustion are not
  forecast. This is optimistic reservation-only coverage, not correctness.
- Basic protection compares each new contract's marginal effect on pre-existing
  pending contracts in the end-of-day calendar; same-day conflicts also appear in
  coverage. Same-day cancelled commitments remain explicit exclusions and are still
  counted in cancellation/financial metrics.
- Capital productivity: 100 × service contribution / deployed-and-inbound capital
  days. Contribution subtracts service expenses, compensation and owned-use wear,
  not fixed daily fees or purchase/sale flows. Trial use fees are charged once.
  Extra-day wear is charged on recorded `not_returned` days through the cutoff,
  rather than using final contract duration (which can include future days).
- Quick cancellations occur 0–2 days after confirmation. Rate denominators require
  a full two-day observation window. Basic external-change evidence is incomplete;
  replay adds public late-return and supplier-delay events, plus own-action tags.

## Extended diagnostics

The public-readiness calendar differs from the basic calendar: it projects current
FIFO cleaning jobs, scheduled maintenance/repair, public rental return estimates
and not-yet-overdue purchase ETAs. Unknown overdue supply is excluded; future
turnaround is one day. It does not simulate future shared cleaning contention,
random damage or future life exhaustion. Supplier trials respect known deadlines
and remaining contract-start allowance. Both methods are explicitly named.

- Actual walk-in intrusion is assessed in native batch order. It records newly
  uncovered reservations; subsequent failure is association, not sole causation.
- Persistent gaps count pending-contract days and retain lineage across versions.
  Clearances and recurrences have candidate source labels, not invented unique
  causes. Cancellation, service completion, pickup and censoring are distinct.
- Action protection is reported by action type with affected and removed old IDs.
- Request opportunities enumerate public profiles and dates, at most 256 complete
  candidates per date. A reached limit with no witness is `undetermined`.
  Fixed service offers cannot be mixed; pre-contract optional omission is allowed
  only for inquiry menus. Batch decisions share the actual pre-call state and are
  labelled accordingly; opportunities cannot all be summed into achievable sales.
- Authorized-choice evaluation includes explicit, fixed-offer, single-plan auto
  and end-of-day default confirmations. Technical feasibility is not willingness
  to pay or optimal profitability.
- Capacity relaxation evaluates days 5,10,... within the decision horizon.
  For each SKU with existing limits, raise each existing limit by one from today
  through today+14, clipped to the legal horizon. Unlimited dates remain unlimited.
  Each candidate uses the real tool and a full native same-day demand replay.
  Reservations, physical stock and prices are never released or changed.
  Contribution uses agreed rent minus normal cleaning and use wear/trial fee,
  never hidden future damage/weather/late-rent outcomes. Increased occupancy is
  checked against currently available stock and old promises at gaps 1/3/7.
  Keep-baseline is always available; zero improvement is not an optimality proof.
  Candidate and independent-date improvements are not additive season profits.
  `--sample-days` changes the protocol label to `custom_diagnostic`.
- Structured object exposure only records exact IDs from prior successful tool
  results and opening reports. It does not parse notes or claim comprehension.

## Integrity and resumption

The evaluator backs up SQLite read-only, including WAL content, then works on copies.
Replay reproduces RunStore's sorted JSON reload boundaries and checks every tool
receipt (excluding existing model-usage-only fields). The baseline demand branch
must reproduce actual rental contracts exactly. A mismatch saves evidence, retains
basic results, returns failure, and is never scored as zero opportunity.

Rejected actor calls, including malformed argument payloads, are still executed
and their rejection receipts compared. They remain in the basic tool-error
statistics, but do not enter opportunity or authorization diagnostics as actions
that took effect. A later valid retry is evaluated normally; an unanswered
request is still evaluated at expiry. Diagnostic parsing never repairs or accepts
an invalid actor submission on the actor's behalf.

Daily low-compression gzip checkpoints and per-candidate results are cached under a
digest of the input snapshot, runtime data, scoring source and sampling protocol.
The shared `replay-cache` is independent of report cutoffs. `--through-days 100 214`
replays once and freezes the Day100 result before observing later events. Repeating
or extending a command on the same input snapshot reuses matching work. A changed
run snapshot or scoring source invalidates the cache. A newly requested earlier
cutoff with no saved result requires replay rather than hindsight reconstruction.

`--workers` limits concurrent runs; `--cpu-budget` limits CPU tasks across ALL runs
and sampled-day subprocesses (default min(8, available CPUs)). Independent sampled
days run concurrently; tools within a trajectory remain sequential. Main workers
release their CPU leases when waiting for branches, including at budget=1. Each
branch uses matching frozen source and checks its persisted no-op against the
original demand receipt. Output order is deterministic. Numerical library threads
are limited to one. Parallelism is not part of metric/cache semantics.

### Pure-query state reuse

`--state-reload reuse` (default) avoids redundant reconstruction after successful,
audited pure queries. It does NOT skip tool execution, receipt comparisons, or
business decisions. The whitelist is pinned to exact V53/V54 core source hashes;
unreviewed source identities automatically use full reconstruction. Each query
name/view/mode variant is checked on first use with an order-sensitive digest of
ALL dataclass fields, including fields omitted by compatibility serialization.
The canonical reload must pass an initial fixed-point check. An unexpected query
mutation fails closed; it is not accepted as zero risk or silently ignored.

Non-whitelisted calls and all failures invalidate reuse. Morning inbox consumption
invalidates reuse without adding or moving the original first-tool boundary.
Restarts always begin with a full reconstruction; in-memory certificates are not
trusted across checkpoints. Unknown queries are conservative even if declared
`read_only` by their schema.

`--state-reload reference` retains full reconstruction before every recorded call.
`--state-reload audit` additionally checks every whitelisted query for mutations
and compares every reused boundary against canonical reconstruction. Audit is an
expensive correctness mode, not a performance mode. All three modes have separate
replay caches and report the selected mode under `execution`; this does not change
scoring/comparison groups. Progress/timing files include reconstruction/reuse
counts, purity checks and conservative invalidation reasons.

Check `replay-cache/KEY/progress.json` for replay day, submitted/completed branches
and phase timings. Reports are complete only after every required branch passes.
An exclusive run-workspace lock prevents duplicate writers. Earlier reports are
preserved under `day-N/history/KEY/` before replacement. The explicit Python API
`legacy_cache=...` only accepts the allowlisted pre-optimization implementation,
matching inputs and complete daily boundaries; it does not ignore version checks.

No API keys are required. Long validations may be detached; inspect `progress.json`
and the saved completion report after they finish. Never rank differing environment
versions/tracks together. Keep sample counts, N/A statuses and business scale visible.
