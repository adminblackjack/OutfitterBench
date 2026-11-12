from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from . import FORMAT

PARAMETERS = {
    "format": FORMAT, "calendar_gaps": [0, 1, 2], "quick_cancel_days": 2,
    "candidate_limit_per_date": 256, "opportunity_gap": 1,
    "relaxation_gaps": [1, 3, 7], "relaxation_sample_stride": 5,
    "relaxation_window_days": 14, "relaxation_delta": 1,
    "positive_improvement_cny": 0.01,
}
FAILURES = {"cancelled_by_manager", "pickup_failed", "merchant_induced_customer_cancellation"}
CONFIRM = "reservation_confirmed_from_customer_plan_set"
OMITTED_LINE_STATUSES = {'omitted','cancelled','optional_removed_at_confirmation',
                         'optional_removed_at_quote','optional_missing_credited'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_digest():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))})


def ratio(numerator, denominator, **extra):
    return dict(value=100*numerator/denominator if denominator else None,
                numerator=numerator, denominator=denominator, unit="percent",
                status="available" if denominator else "not_applicable", **extra)


def unavailable(reason="not_enabled"):
    return dict(value=None, status=reason)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                        if isinstance(v, (list, dict)) else v for k, v in row.items()})


def line_units(lines):
    """Committed optional lines count; pre-contract omitted lines do not."""
    out = Counter()
    for line in lines:
        if line.get('disposition') == 'omit' or line.get('status') in OMITTED_LINE_STATUSES:
            continue
        quantity = line.get('asset_quantity')
        if quantity is None:
            quantity = line.get('quantity')
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise ValueError('missing physical quantity in selected plan')
        oid = line.get('selected_offer_id') or line.get('offer_id')
        if not oid:
            raise ValueError('missing exact offer in selected plan')
        out[oid] += quantity
    return dict(out)


def occupied(record, day, gap):
    return {(oid, t): qty for oid, qty in record['units'].items()
            for t in range(max(day, record['start']), record['end']+gap)}


def loads(book, day, gap):
    result = Counter()
    for record in book.values():
        result.update(occupied(record, day, gap))
    return result


def deficits(record, total, supply, day, gap):
    return [dict(offer_id=oid, day=t, demand=total[oid, t], supply=supply(oid, t),
                 shortage=total[oid, t]-supply(oid, t))
            for oid, t in occupied(record, day, gap) if total[oid, t] > supply(oid, t)]


def feasible(record, total_without, supply, day, gap):
    return all(total_without[key]+qty <= supply(*key)
               for key, qty in occupied(record, day, gap).items())
