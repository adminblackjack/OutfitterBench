"""Lightweight, model-independent views of existing offline evaluation reports.

This module deliberately lives outside evaluation/: adding a presentation layer
must not invalidate frozen evaluator identities or expensive replay caches.
No simulator, database, provider, or replay is invoked here.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time


FORMAT = "outfitter-behavioral-diagnostics-v1"
SOURCE_FORMAT = "outfitter-trajectory-metrics-v1"
LEAD_BANDS = (("1-2", 1, 2), ("3-7", 3, 7), ("8+", 8, None))
ADMISSION_ACTIONS = frozenset({"quote_reservation_inquiries", "confirm_reservation_plans",
                               "respond_reservation_service_offers"})
ADJUSTMENT_ACTIONS = frozenset({"respond_reservation_changes", "apply_reservation_contingency",
                                "sell_to_liquidation_dealer", "schedule_maintenance",
                                "repair_or_retire_assets"})
CORE_LABELS = {
    "pending_booking_gap_rate": "Pending-booking gap rate",
    "near_pickup_gap_rate": "Gap rate 1-2 days before pickup",
    "walk_in_induced_gap_rate": "Walk-in-induced new gap rate",
    "capital_productivity": "Capital productivity",
    "capacity_positive_day_rate_g3": "Local capacity-improvement frequency (g=3)",
    "capacity_mean_improvement_g3": "Mean local capacity improvement (g=3)",
}


class DiagnosticError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def code_identity():
    files = [Path(__file__), Path(__file__).with_name("behavioral_diagnostic_plots.py")]
    return {p.name: file_digest(p) for p in files}


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "expected a finite number")
    return value


def count(value):
    require(type(value) is int and value >= 0, "expected a nonnegative integer count")
    return value


def ratio(n, d):
    count(n); count(d)
    require(n <= d, "numerator exceeds denominator")
    return {"value": 100*n/d if d else None, "numerator": n, "denominator": d,
            "unit": "percent", "status": "available" if d else "not_applicable"}


def unavailable(status, unit="percent"):
    return {"value": None, "numerator": None, "denominator": None,
            "unit": unit, "status": status}


def checked_metric(obj, *, percent=True):
    require(isinstance(obj, dict), "missing metric")
    value, n, d = obj.get("value"), obj.get("numerator"), obj.get("denominator")
    if percent:
        expected = ratio(count(n), count(d))
    else:
        number(n); number(d)
        require(d >= 0, "negative capital-day denominator")
        expected = {"value": 100*n/d if d else None,
                    "status": "available" if d else "not_applicable"}
    require(obj.get("status") == expected["status"], "inconsistent metric status")
    if expected["value"] is None:
        require(value is None, "undefined metric must be null")
    else:
        require(math.isclose(number(value), expected["value"], rel_tol=1e-10, abs_tol=1e-8),
                "metric value does not match numerator/denominator")
    if percent:
        require(obj.get("unit") == "percent", "unexpected rate unit")
    else:
        require(obj.get("unit") == "CNY per 100 capital-CNY-days", "unexpected capital unit")
    return copy.deepcopy(obj)


def verify_rate(obj, n, d):
    checked_metric(obj)
    require(obj["numerator"] == n and obj["denominator"] == d,
            "stored rate differs from object-level records")
    return ratio(n, d)


def derive_report(report, source_sha256, derivation_code):
    """Pure transformation; identical source reports always give identical values."""
    require(report.get("format") == SOURCE_FORMAT, "unsupported source evaluation format")
    day = count(report.get("through_day"))
    require(day > 0, "cutoff must be positive")
    identity, protocol = report["input_identity"], report["protocol"]
    require(identity.get("through_day") == day and identity.get("parameters") == protocol,
            "cutoff or protocol identity mismatch")
    require(report.get("cache_key") == digest(identity), "source identity checksum mismatch")
    require(report.get("sample_protocol") in {"default", "custom_diagnostic"},
            "unknown sample protocol")
    basic, ext = report["basic"], report.get("extended", {})
    ext_status = ext.get("status", "not_enabled")
    require(ext_status in {"complete", "not_enabled"},
            f"incomplete or failed extended evaluation: {ext_status}")
    service, business = basic["service"], basic["business"]
    reliability = checked_metric(service["reliability"])
    verify_rate(reliability, count(service["pickups"]),
                service["pickups"] + count(service["merchant_failures"]))
    capital = checked_metric(basic["capital_productivity"], percent=False)
    final = business.get("final_score_available") is True
    score = business.get("evaluation_score") if final else None
    net = business.get("value_change") if final and not business.get("run_bankrupt_at_snapshot") else None
    if score is not None:
        number(score)
    if net is not None:
        require(math.isclose(number(net), score-number(business["initial_value"]), abs_tol=.011),
                "business value change does not reconcile")
    result = {
        "format": FORMAT, "run_id": report["run_id"], "through_day": day,
        "run_metadata": copy.deepcopy(report["run_metadata"]),
        "source": {"report_sha256": source_sha256, "format": SOURCE_FORMAT,
                   "comparison_group": report["comparison_group"],
                   "sample_protocol": report["sample_protocol"],
                   "input_identity": copy.deepcopy(identity), "protocol": copy.deepcopy(protocol),
                   "environment_identity": copy.deepcopy(report["environment_identity"])},
        "derivation_code": derivation_code, "extended_status": ext_status,
        "outcomes": {"official_score": score, "net_value": net,
                     "terminal_score_available": final, "pickups": service["pickups"],
                     "reliability": reliability, "merchant_failures": service["merchant_failures"],
                     "customer_cancellations": service["customer_cancellations"],
                     "pending": service["pending"]},
        "core": {key: unavailable(ext_status) for key in CORE_LABELS},
        "gap_by_lead": {label: unavailable(ext_status) for label, _, _ in LEAD_BANDS},
        "action_new_gaps": {"status": ext_status, "by_action": {}, "events": []},
        "capacity_relaxation": {},
        "support": {"nominal_coverage": {g: checked_metric(v["coverage"])
                    for g, v in basic["nominal_calendar"].get("metrics", {}).items()}},
        "warnings": list(report.get("warnings", [])),
    }
    result["core"]["capital_productivity"] = capital
    result["core"]["capacity_mean_improvement_g3"] = unavailable(ext_status, "CNY per sampled day")
    if ext_status != "complete":
        return result
    require(ext.get("method") == "public-readiness-calendar-v1", "unknown readiness projection")
    risk = ext["persistent_gaps"]["records"]
    seen = set()
    for row in risk:
        key = row["commitment_id"], count(row["day"])
        require(key not in seen and 1 <= row["day"] <= day, "duplicate or out-of-cutoff booking-day")
        seen.add(key)
        require(type(row["gap"]) is bool and count(row["lead_days"]) > 0,
                "invalid future-booking observation")
        require(row["pickup_day"] - row["day"] == row["lead_days"], "incorrect pickup lead")
        require(row["gap"] == bool(row["shortages"]), "gap flag disagrees with shortages")
    result["core"]["pending_booking_gap_rate"] = verify_rate(
        ext["persistent_gaps"]["metric"], sum(r["gap"] for r in risk), len(risk))
    for label, low, high in LEAD_BANDS:
        subset = [r for r in risk if low <= r["lead_days"] and (high is None or r["lead_days"] <= high)]
        result["gap_by_lead"][label] = verify_rate(
            ext["persistent_gaps"]["by_lead"][label], sum(r["gap"] for r in subset), len(subset))
    result["core"]["near_pickup_gap_rate"] = copy.deepcopy(result["gap_by_lead"]["1-2"])
    walk = ext["walk_in_intrusion"]["records"]
    require(all(1 <= count(r["day"]) <= day for r in walk), "walk-in observation outside cutoff")
    require(len({(r["day"], r["batch_id"]) for r in walk}) == len(walk), "duplicate walk-in batch")
    result["core"]["walk_in_induced_gap_rate"] = verify_rate(
        ext["walk_in_intrusion"]["metric"], sum(bool(r["affected_prior"]) for r in walk), len(walk))
    actions = ext["action_protection"]["records"]
    require(all(1 <= count(r["day"]) <= day for r in actions), "action observation outside cutoff")
    require(set(ext["action_protection"]["by_action"]) == {r["tool"] for r in actions},
            "action groups do not match records")
    for tool, source_rate in sorted(ext["action_protection"]["by_action"].items()):
        rows = [r for r in actions if r["tool"] == tool and r["relevant_prior"]]
        for r in rows:
            require(set(r["affected_prior"]) <= set(r["relevant_prior"]), "affected action lacks exposure")
            require(not set(r["affected_prior"]) & set(r["removed_prior"]),
                    "removed commitment counted as newly gapped")
        n = sum(bool(r["affected_prior"]) for r in rows)
        verify_rate(source_rate, len(rows)-n, len(rows))
        result["action_new_gaps"]["by_action"][tool] = dict(
            ratio(n, len(rows)), group="admission" if tool in ADMISSION_ACTIONS else
            "adjustment" if tool in ADJUSTMENT_ACTIONS else "other",
            affected_unique_commitments=len({cid for r in rows for cid in r["affected_prior"]}),
            removal_calls=sum(bool(r["removed_prior"]) for r in rows),
            removed_unique_commitments=len({cid for r in rows for cid in r["removed_prior"]}),
            relevant_removal_only_calls=sum(bool(r["removed_prior"]) and not r["affected_prior"] for r in rows))
    for index, r in enumerate(actions):
        if r["affected_prior"] or r["removed_prior"]:
            result["action_new_gaps"]["events"].append(dict(
                source_record_index=index, **copy.deepcopy(r),
                introduced_gap=bool(r["affected_prior"]),
                interpretation="removal is reported separately, not a newly gapped surviving commitment"))
    # Record the legacy exposure denominator, rather than silently substituting
    # a new denominator or interpreting every no-new-gap call as successful care.
    result["action_new_gaps"]["denominator_definition"] = "supported successful calls touching resources of prior bookings"
    relax = ext["capacity_relaxation"]
    records = relax["records"]
    sample_days = identity["sample_days"]
    require(len(sample_days) == len(set(sample_days)) and sorted(r["day"] for r in records) == sample_days,
            "capacity sample dates mismatch")
    require(all(1 <= d <= min(day, report["run_metadata"]["max_days"]) for d in sample_days),
            "sample outside completed decision days")
    if report["sample_protocol"] == "default":
        require(sample_days == list(range(5, min(day, report["run_metadata"]["max_days"])+1, 5)),
                "default sampling must retain every fifth decision day")
    require(protocol["relaxation_gaps"] == [1, 3, 7] and
            protocol["positive_improvement_cny"] == .01 and
            protocol["relaxation_delta"] == 1 and protocol["relaxation_window_days"] == 14,
            "unsupported local-capacity protocol")
    for gap in ("1", "3", "7"):
        values = [number(r["improvements"][gap]) for r in records]
        require(all(v >= 0 for v in values), "negative best local improvement")
        require(all(count(r["candidate_count"]) > 0 or r["improvements"][gap] == 0 for r in records),
                "positive improvement without a candidate")
        n, d = sum(v >= .01 for v in values), len(values)
        mean = sum(values)/d if d else None
        source = relax["by_gap"][gap]
        require(source["evaluated_days"] == d and source["positive_days"] == n,
                "local-capacity counts mismatch")
        require((mean is None and source["mean_daily_improvement"] is None) or
                (mean is not None and math.isclose(mean, number(source["mean_daily_improvement"]),
                                                 abs_tol=1e-9, rel_tol=1e-10)),
                "local-capacity mean mismatch")
        absent = sum(r["candidate_count"] == 0 for r in records)
        require(source["no_candidate_days"] == absent, "candidate support mismatch")
        result["capacity_relaxation"][gap] = {
            "positive_day_rate": ratio(n, d),
            "mean_improvement": {"value": mean, "numerator": sum(values), "denominator": d,
                "unit": "CNY per sampled day", "status": "available" if d else "not_applicable"},
            "no_candidate_days": absent, "sample_days": list(sample_days),
            "passed_candidates": source["passed_candidates"],
        }
    result["core"]["capacity_positive_day_rate_g3"] = copy.deepcopy(result["capacity_relaxation"]["3"]["positive_day_rate"])
    result["core"]["capacity_mean_improvement_g3"] = copy.deepcopy(result["capacity_relaxation"]["3"]["mean_improvement"])
    for field in ("opportunities", "authorization_selection"):
        result["support"][field] = checked_metric(ext[field]["metric"])
    return result


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                         if isinstance(row.get(key), (list, dict)) else row.get(key)
                         for key in fields})
    return stream.getvalue()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n"


def display(value, places=2):
    return "N/A" if value is None else f"{value:.{places}f}"


def metric_rows(profile):
    return [dict(metric=name, **obj) for name, obj in profile["core"].items()]


def action_rows(profile):
    return [dict(action=tool, **obj) for tool, obj in profile["action_new_gaps"]["by_action"].items()]


def report_text(profile):
    lines = ["# Outfitter behavioral diagnostics", "",
             f"Run: `{profile['run_id']}`; completed-day cutoff: {profile['through_day']}.",
             f"Derived format: `{FORMAT}`; source extended status: `{profile['extended_status']}`.", "",
             "## Outcomes", "",
             f"- Official score: {display(profile['outcomes']['official_score'])} CNY.",
             f"- Net business value change: {display(profile['outcomes']['net_value'])} CNY.",
             f"- Deliveries: {profile['outcomes']['pickups']}; reliability: "
             f"{display(profile['outcomes']['reliability']['value'])}%.", "",
             "## Core diagnostics", "",
             "| Diagnostic | Value | Numerator | Denominator | Unit | Status |",
             "|---|---:|---:|---:|---|---|"]
    for row in metric_rows(profile):
        lines.append(f"| {CORE_LABELS[row['metric']]} | {display(row['value'], 4)} | "
                     f"{row.get('numerator')} | {row.get('denominator')} | {row['unit']} | {row['status']} |")
    lines += ["", "## Gap rates by pickup lead", "",
              "| Lead | Gap booking-days | Observed booking-days | Rate (%) | Status |",
              "|---|---:|---:|---:|---|"]
    for label, _, _ in LEAD_BANDS:
        obj = profile["gap_by_lead"][label]
        lines.append(f"| {label} | {obj['numerator']} | {obj['denominator']} | {display(obj['value'])} | {obj['status']} |")
    lines += ["", "## New gaps introduced by subsequent actions", "",
              "| Action | Group | Gap calls / relevant calls | Rate (%) | Removal calls | Status |",
              "|---|---|---:|---:|---:|---|"]
    for row in action_rows(profile):
        lines.append(f"| {row['action']} | {row['group']} | {row['numerator']}/{row['denominator']} | "
                     f"{display(row['value'])} | {row['removal_calls']} | {row['status']} |")
    if not profile["action_new_gaps"]["by_action"]:
        lines += [f"| Not computed | | | | | {profile['extended_status']} |"]
    lines += ["", "## Interpretation", "",
              "- These are complementary diagnostics, not a composite or independent pure-ability score.",
              "- Pickup-lead groups are different observation sets, not a fixed contract cohort's recovery curve.",
              "- Gap exposure counts booking-days: a booking observed on three days contributes three observations, not three failures.",
              "- Action rates use supported successful tool calls touching prior booking resources; walk-ins use customer-trip batches. Do not pool their denominators.",
              "- A removal is not counted as a new gap in a surviving contract, and is not credited as successful protection. Removal calls and IDs remain separate evidence.",
              "- Existing deficits becoming worse are outside the newly-introduced-gap numerator. Unsupported action types remain outside the source evaluation.",
              "- Local improvement is an estimate under fixed checks, not realized revenue or additive recoverable season profit. Low local opportunity alone does not establish good protection.",
              "- No-candidate and zero-improvement dates remain in the local mean; missing evaluation is never a zero.",
              "- Nominal/option/authorized-choice coverage remains supporting data. Quick cancellation and pairwise rents are not core profile scores.",
              "- All computations read an existing evaluator snapshot; no fresh replay or current-state re-evaluation occurred."]
    lines += ["", "## Source", "", f"Report SHA-256: `{profile['source']['report_sha256']}`.",
              f"Source comparison group: `{profile['source']['comparison_group']}`.",
              f"Sampling: `{profile['source']['sample_protocol']}`; full provenance is in `profile.json`."]
    if profile["warnings"]:
        lines += ["", "## Source warnings", "", *[f"- {w}" for w in profile["warnings"]]]
    return "\n".join(lines)+"\n"


def publish_files(directory, contents):
    """Publish validated derived files via unique temporary files; never input files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, text in contents.items():
        target = directory/name
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".diagnostics-", delete=False) as temporary:
            temporary.write(text)
            temp_path = Path(temporary.name)
        os.replace(temp_path, target)
        hashes[name] = file_digest(target)
    return hashes


def profile_files(profile):
    fields = ["metric", "value", "numerator", "denominator", "unit", "status"]
    leads = [dict(lead=label, **obj) for label, obj in profile["gap_by_lead"].items()]
    local = [dict(gap=g, positive_day_rate=obj["positive_day_rate"]["value"],
                  positive_days=obj["positive_day_rate"]["numerator"],
                  evaluated_days=obj["positive_day_rate"]["denominator"],
                  mean_improvement=obj["mean_improvement"]["value"],
                  status=obj["positive_day_rate"]["status"],
                  no_candidate_days=obj["no_candidate_days"], sample_days=obj["sample_days"])
             for g, obj in profile["capacity_relaxation"].items()]
    return {
        "profile.json": json_text(profile), "REPORT.md": report_text(profile),
        "metrics.csv": csv_text(metric_rows(profile), fields),
        "gap_by_lead.csv": csv_text(leads, ["lead", *fields[1:]]),
        "action_new_gaps.csv": csv_text(action_rows(profile), ["action", "group", *fields[1:],
            "affected_unique_commitments", "removal_calls", "removed_unique_commitments", "relevant_removal_only_calls"]),
        "action_gap_events.csv": csv_text(profile["action_new_gaps"]["events"], ["source_record_index", "day", "tool",
            "relevant_prior", "affected_prior", "removed_prior", "introduced_gap", "shortages"]),
        "capacity_relaxation.csv": csv_text(local, ["gap", "positive_day_rate", "positive_days", "evaluated_days",
            "mean_improvement", "status", "no_candidate_days", "sample_days"]),
    }


def diagnose_run(run_dir, *, through_day=214, publish=True):
    started = time.monotonic()
    root = Path(run_dir).expanduser().resolve()
    require(type(through_day) is int and through_day > 0, "cutoff must be a positive integer")
    source = root/"analysis/evaluation"/SOURCE_FORMAT/f"day-{through_day}/metrics.json"
    manifest_path = root/"manifest.json"
    try:
        raw, manifest_raw = source.read_bytes(), manifest_path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
        report, manifest = json.loads(raw), json.loads(manifest_raw)
        require(report["through_day"] == through_day, "requested/report cutoff mismatch")
        require(report["run_id"] == manifest["run_id"], "run identity mismatch")
        require(report["run_metadata"]["seed"] == manifest["config"]["seed"], "seed identity mismatch")
        normalized = dict(manifest, runtime_data="data")
        require(report["input_identity"]["manifest"] == digest(normalized), "manifest no longer matches evaluator snapshot")
        profile = derive_report(report, source_hash, code_identity())
        profile["source"]["report_path"] = str(source)
        profile["source"]["manifest_sha256"] = manifest_hash
        require(file_digest(source) == source_hash and file_digest(manifest_path) == manifest_hash,
                "source changed while deriving diagnostics")
        require(profile["derivation_code"] == code_identity(), "diagnostic code changed during derivation")
        destination = root/"analysis/evaluation"/FORMAT/f"day-{through_day}"
        hashes = publish_files(destination, profile_files(profile)) if publish else {}
        return {"run_dir": str(root), "output_dir": str(destination), "profile": profile,
                "outputs_sha256": hashes, "worker_pid": os.getpid(),
                "elapsed_seconds": round(time.monotonic()-started, 6)}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise DiagnosticError(f"{root.name}: {error}") from error


def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def worker_count(workers, tasks):
    require(workers is None or (type(workers) is int and workers > 0), "workers must be positive")
    return min(tasks, cpu_count(), workers or cpu_count()) if tasks else 0


def worker_environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"


@contextmanager
def parallel_environment():
    """Set thread limits before spawn imports NumPy, then restore the caller."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    worker_environment()
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _task(job):
    root, cutoff, publish = job
    return diagnose_run(root, through_day=cutoff, publish=publish)


def diagnose_many(run_dirs, *, through_day=214, workers=None, publish=True):
    roots = [str(Path(p).expanduser().resolve()) for p in run_dirs]
    require(bool(roots) and len(set(roots)) == len(roots), "empty or duplicate run directories")
    jobs = [(r, through_day, publish) for r in roots]
    concurrency = worker_count(workers, len(jobs))
    if concurrency == 1:
        return [_task(job) for job in jobs]
    ordered = {}
    with parallel_environment(), ProcessPoolExecutor(max_workers=concurrency, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_environment) as pool:
        futures = {pool.submit(_task, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    return [ordered[i] for i in range(len(jobs))]


def grouped_profiles(results, presentation=None, comparison_policy=None):
    """Group raw protocols strictly; an explicit validated study may reconcile budgets."""
    presentation = presentation or {}
    groups = defaultdict(list)
    for result in results:
        p = result["profile"]
        context = presentation.get(result["run_dir"], {})
        raw_group = p["source"]["comparison_group"]
        group = raw_group
        if context.get("comparison_group"):
            require(comparison_policy and context.get("source_sha256") == p["source"]["report_sha256"],
                    "comparison reconciliation requires a validated policy bound to the exact report")
            group = context["comparison_group"]
        # Sampling and cutoff never disappear under a presentation policy.
        key = digest({"group": group, "day": p["through_day"], "protocol": p["source"]["protocol"],
                      "sampling": p["source"]["sample_protocol"],
                      "sample_days": p["source"]["input_identity"]["sample_days"]})
        model = context.get("model", p["run_metadata"].get("model", "unknown"))
        effort = p["run_metadata"].get("reasoning_effort")
        groups[key].append({"id": context.get("id", p["run_id"]), "model": model,
            "model_key": digest([model, effort]), "label": context.get("label", str(model)),
            "reasoning_effort": effort, "order": context.get("order", str(model)),
            "color": context.get("color"), "marker": context.get("marker", "o"),
            "axis_label": context.get("axis_label", context.get("label", str(model))),
            "seed": p["run_metadata"]["seed"], "run_dir": result["run_dir"],
            "source_comparison_group": raw_group, "profile": p})
    for members in groups.values():
        require(len({(m["run_dir"], m["profile"]["through_day"]) for m in members}) == len(members),
                "duplicate trajectory in cohort")
        members.sort(key=lambda m: (m["seed"], str(m["order"]), m["id"]))
    return dict(sorted(groups.items()))


def aggregate_members(members):
    """Within-seed means, then seed-equal means; never pool denominators."""
    models = defaultdict(list)
    for row in members:
        models[row["model_key"]].append(row)
    output = []
    for _, rows in sorted(models.items()):
        metrics = {}
        for name in CORE_LABELS:
            seeds = defaultdict(list)
            for row in rows:
                value = row["profile"]["core"][name]["value"]
                if value is not None:
                    seeds[row["seed"]].append(value)
            means = {str(seed): sum(values)/len(values) for seed, values in sorted(seeds.items())}
            metrics[name] = {"value": sum(means.values())/len(means) if means else None,
                             "per_seed": means, "contributing_seeds": len(means),
                             "contributing_runs": sum(len(v) for v in seeds.values()),
                             "registered_seeds": sorted({r["seed"] for r in rows})}
        output.append({"model": rows[0]["model"], "label": rows[0]["label"],
                       "reasoning_effort": rows[0]["reasoning_effort"], "metrics": metrics})
    return output


def write_cohort(results, output_dir, *, presentation=None, comparison_policy=None, workers=None, plots=True):
    root = Path(output_dir).expanduser().resolve()
    require(bool(results), "empty cohort")
    require(all(r["profile"]["derivation_code"] == code_identity() for r in results),
            "mixed or changed diagnostic derivation code")
    require(all(not root.is_relative_to(Path(r["run_dir"])) for r in results),
            "cohort output must be separate from source run directories")
    groups = grouped_profiles(results, presentation, comparison_policy)
    # A failed re-render must not leave an earlier completion marker claiming
    # that a partially replaced set of derived files is complete.
    publish_files(root, {"COMPLETE.json": json_text({"status": "building", "runs": len(results)})})
    manifest = {"format": FORMAT, "comparison_policy": comparison_policy,
                "derivation_code": code_identity(), "groups": {}, "execution": {
                    "available_cpus": cpu_count(), "requested_workers": workers,
                    "workers_used": worker_count(workers, len(results)),
                    "worker_pids": sorted({r["worker_pid"] for r in results}),
                    "run_seconds": {r["profile"]["run_id"]: r["elapsed_seconds"] for r in results}},
                "sources": {r["run_dir"]: r["profile"]["source"] for r in results}}
    all_lines = ["# Outfitter auxiliary diagnostics", "",
                 "Profiles are derived from existing evaluations; no model calls or replay were run.",
                 "Raw groups are retained. Any explicit study reconciliation is recorded in manifest.json.", ""]
    plot_jobs = []
    for key, members in groups.items():
        directory = root if len(groups) == 1 else root/key[:16]
        summary, action_data, lead_data, capacity_data = [], [], [], []
        lines = ["# Same-seed operating profiles", "",
                 f"Completed-day cutoff: {members[0]['profile']['through_day']}; sampling: {members[0]['profile']['source']['sample_protocol']}.", "",
                 "Rates are percentages; capital is CNY per 100 capital-CNY-days; local gain is CNY per sampled day.",
                 "Values are not normalized against other models. No auxiliary total score is defined."]
        if comparison_policy:
            lines += ["", "Comparison uses the validated study registry. Declared execution overrides: "
                      + json.dumps(comparison_policy.get("execution_config_overrides", {}), sort_keys=True)
                      + ". Original comparison groups and the validation policy are retained in manifest.json."]
        for seed in sorted({m["seed"] for m in members}):
            lines += ["", f"## Seed {seed}", "",
                      "| Model | Net value | Delivered | Reliability | Gap | Near pickup | Capital | Local positive | Local gain |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for m in (x for x in members if x["seed"] == seed):
                p, core, out = m["profile"], m["profile"]["core"], m["profile"]["outcomes"]
                values = [out["net_value"], out["pickups"], out["reliability"]["value"],
                          *[core[k]["value"] for k in ("pending_booking_gap_rate", "near_pickup_gap_rate",
                            "capital_productivity", "capacity_positive_day_rate_g3", "capacity_mean_improvement_g3")]]
                places = (2, 0, 2, 2, 2, 3, 1, 3)
                lines.append("| "+m["label"]+" | "+" | ".join(display(v, n) for v, n in zip(values, places))+" |")
                row = {"run": m["id"], "model": m["label"], "seed": seed, **{k: out[k] for k in ("official_score", "net_value", "pickups")},
                       "reliability": out["reliability"]["value"], "extended_status": p["extended_status"]}
                for name, obj in core.items():
                    row[name] = obj["value"]
                    row[name+"_status"] = obj["status"]
                    row[name+"_numerator"] = obj.get("numerator")
                    row[name+"_denominator"] = obj.get("denominator")
                summary.append(row)
                for act in action_rows(p):
                    action_data.append(dict(run=m["id"], seed=seed, model=m["label"], **act))
                action_data.append(dict(run=m["id"], seed=seed, model=m["label"], action="walk_in",
                                        group="walk_in_batch", **core["walk_in_induced_gap_rate"]))
                lead_data.extend(dict(run=m["id"], seed=seed, model=m["label"], lead=label, **obj)
                                 for label, obj in p["gap_by_lead"].items())
                for gap, obj in p["capacity_relaxation"].items():
                    capacity_data.append(dict(run=m["id"], seed=seed, gap=gap,
                        positive_days=obj["positive_day_rate"]["numerator"],
                        evaluated_days=obj["positive_day_rate"]["denominator"],
                        rate=obj["positive_day_rate"]["value"], mean_improvement=obj["mean_improvement"]["value"],
                        no_candidate_days=obj["no_candidate_days"]))
        action_lines = ["# Action-induced new gaps", "", "Tool calls and walk-in batches are separate exposure units. Removals are not credited as successful protection."]
        for seed in sorted({m["seed"] for m in members}):
            action_lines += ["", f"## Seed {seed}", "", "| Run | Action | New gap / relevant | Rate (%) | Removal calls | Status |", "|---|---|---:|---:|---:|---|"]
            for row in (x for x in action_data if x["seed"] == seed):
                action_lines.append(f"| {row['run']} | {row['action']} | {row.get('numerator')}/{row.get('denominator')} | "
                                    f"{display(row['value'])} | {row.get('removal_calls','')} | {row['status']} |")
        lines += ["", "## Reading the profiles", "",
                  "- N/A is not zero or perfect performance; per-run statuses distinguish uncomputed metrics from empty exposure.",
                  "- Gap exposure counts booking-days, not distinct failed bookings.",
                  "- Lead groups are not a fixed cohort recovery curve. Preserve delivery scale and cancellation outcomes when interpreting gaps.",
                  "- Low local opportunity can coexist with high shortage rates. Local gains are estimates, not additive season profit.",
                  "- See action_new_gaps.md for source-specific call/batch counts; no pooled action rate is used.",
                  "- Optional/nominal/authorized-choice coverage remains in each profile's supporting data."]
        missing = sorted({m["label"] for m in members if m["profile"]["extended_status"] != "complete"})
        if missing:
            lines += ["- Extended evaluation not computed for: "+", ".join(missing)+"."]
        contents = {"SUMMARY.md": "\n".join(lines)+"\n", "profiles.json": json_text(members),
                    "seed_equal_means.json": json_text(aggregate_members(members)),
                    "core_metrics.csv": csv_text(summary, list(summary[0])),
                    "action_new_gaps.md": "\n".join(action_lines)+"\n",
                    "action_new_gaps.csv": csv_text(action_data, ["run", "model", "seed", "action", "group", "value", "numerator", "denominator", "status", "affected_unique_commitments", "removal_calls", "removed_unique_commitments"]),
                    "gap_by_lead.csv": csv_text(lead_data, ["run", "model", "seed", "lead", "value", "numerator", "denominator", "status"]),
                    "capacity_relaxation.csv": csv_text(capacity_data, ["run", "seed", "gap", "positive_days", "evaluated_days", "rate", "mean_improvement", "no_candidate_days"])}
        outputs = publish_files(directory, contents)
        manifest["groups"][key] = {"directory": str(directory), "runs": [m["id"] for m in members], "outputs_sha256": outputs}
        all_lines.append(f"- [{len(members)} runs: summary]({directory.relative_to(root).as_posix()}/SUMMARY.md)")
        relative = directory.relative_to(root).as_posix()
        all_lines.extend(f"  - [{label}]({relative}/{name})" for label, name in (
            ("Core metrics CSV", "core_metrics.csv"), ("Action breakdown", "action_new_gaps.md"),
            ("Seed-equal means", "seed_equal_means.json")))
        if plots:
            plot_jobs.extend((kind, members, str(directory)) for kind in ("operating_profiles", "pickup_lead_gaps", "capacity_opportunities"))
            all_lines.extend(f"  - [{label}]({relative}/{name}.png)" for label, name in (
                ("Operating profiles", "operating_profiles"), ("Pickup-lead gaps", "pickup_lead_gaps"),
                ("Capacity opportunities", "capacity_opportunities")))
    if plot_jobs:
        from .behavioral_diagnostic_plots import render_plot
        with parallel_environment(), ProcessPoolExecutor(max_workers=worker_count(workers, len(plot_jobs)),
                                 mp_context=multiprocessing.get_context("spawn"), initializer=worker_environment) as pool:
            list(pool.map(render_plot, plot_jobs))
    for result in results:
        require(file_digest(result["profile"]["source"]["report_path"]) == result["profile"]["source"]["report_sha256"],
                "source changed while building cohort")
        require(file_digest(Path(result["run_dir"])/"manifest.json") == result["profile"]["source"]["manifest_sha256"],
                "run manifest changed while building cohort")
    for group in manifest["groups"].values():
        directory = Path(group["directory"])
        group["outputs_sha256"].update({p.name: file_digest(p) for p in directory.iterdir()
                                       if p.suffix in {"png", "pdf"} and p.is_file()})
    all_lines += ["", "## Individual reports", "", "| Run | Model | Seed | Extended status | Report |",
                  "|---|---|---:|---|---|"]
    outputs_by_run = {r["run_dir"]: r["output_dir"] for r in results}
    for members in groups.values():
        for m in members:
            target = os.path.relpath(Path(outputs_by_run[m["run_dir"]])/"REPORT.md", root)
            all_lines.append(f"| {m['id']} | {m['label']} | {m['seed']} | "
                             f"{m['profile']['extended_status']} | [Report]({target}) |")
    publish_files(root, {"INDEX.md": "\n".join(all_lines)+"\n", "manifest.json": json_text(manifest),
                         "COMPLETE.json": json_text({"status": "complete", "runs": len(results), "groups": len(groups)})})
    return manifest
