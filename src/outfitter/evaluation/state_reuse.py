"""Version-scoped pure-query reuse; the canonical reference boundary stays intact."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import time

from ..models import WorldState

AUDITED_CORE_SOURCES = frozenset({
    '0b65bb6d6ecb2f616df24e6b368d511e2c80d3492780db3a319bc4ba4d121986',  # V53
    '8e9163d760d1830ffe214f041713a4d9b73ec1490912aacea134a31ba8e44f8f',  # V54
    # V54 release packaging: only call_identity.py differs in the 78-file core.
    # The change parses recorded Responses output; query transitions are unchanged.
    '75bae54ae618f9e28769d1ccddbf77c18109c4cd7ef29c22da3c3c2c5adc143a',
    # Public-weather packaging: runner.py only adds weather_dataset_id to the
    # saved manifest. All state transitions and query implementations match the
    # preceding release; reference/reuse/audit equivalence is regression-tested.
    '846d4cdd58644ea8369af70c785d3cd81b284d65fbd0353385544f247d6de20c',
})
PURE_QUERIES = frozenset({
    'read_business_rules', 'memory_read', 'memory_list', 'memory_search',
    'weather_forecast', 'view_rental_assets', 'view_active_rentals',
    'view_rental_capacity', 'counter_log', 'reservation_inbox',
    'view_reservation', 'view_reservation_quote_workspace', 'business_status',
})
MODES = ('reuse', 'reference', 'audit')


def canonical_reload(state):
    return WorldState.from_dict(json.loads(json.dumps(state.to_dict(), sort_keys=True)))


def ordered_state_digest(state):
    # Do not use to_dict(): its compatibility pruning hides some live fields.
    # Do not sort keys: insertion order is part of the replay contract.
    return hashlib.sha256(json.dumps(asdict(state), ensure_ascii=False,
        sort_keys=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


class QueryStateReuse:
    def __init__(self, mode, source_identity, effects, *, reload=canonical_reload):
        if mode not in MODES:raise ValueError(f'unknown state reload mode: {mode}')
        self.mode=mode; self.reload=reload; self.effects=effects
        self.supported=source_identity.get('source_sha256') in AUDITED_CORE_SOURCES
        self.enabled=mode!='reference' and self.supported
        self.clean=False; self.fixed_point_verified=False
        self.certified=set(); self.pending=None
        self.rebuilt=0; self.reused=0; self.purity_checks=0; self.audit_seconds=0.0
        self.successful_queries=Counter(); self.invalidations=Counter()

    def before_call(self, state):
        if self.enabled and self.clean:
            if self.mode=='audit':
                start=time.monotonic()
                reference=self.reload(state)
                if ordered_state_digest(reference)!=ordered_state_digest(state):
                    raise RuntimeError('pure-query state differs from canonical reference boundary')
                self.audit_seconds+=time.monotonic()-start
            self.reused+=1
            return state
        state=self.reload(state); self.rebuilt+=1
        self.clean=self.enabled
        if self.enabled and not self.fixed_point_verified:
            start=time.monotonic()
            if ordered_state_digest(self.reload(state))!=ordered_state_digest(state):
                raise RuntimeError('canonical reload is not idempotent; query reuse is unsafe')
            self.fixed_point_verified=True
            self.audit_seconds+=time.monotonic()-start
        return state

    def invalidate(self, reason):
        self.clean=False; self.invalidations[reason]+=1

    def before_tool(self, state, name, args):
        self.pending=None
        if not self.enabled or name not in PURE_QUERIES or self.effects.get(name)!='read_only':return
        variant=(name,str(args.get('view','')),str(args.get('mode','')))
        if self.mode=='audit' or variant not in self.certified:
            start=time.monotonic()
            self.pending=(variant,ordered_state_digest(state))
            self.audit_seconds+=time.monotonic()-start

    def after_tool(self, state, name, args, ok):
        if self.pending is not None:
            start=time.monotonic();variant,before=self.pending
            self.pending=None
            if ordered_state_digest(state)!=before:
                raise RuntimeError(f'pure-query whitelist violation: {variant}')
            self.audit_seconds+=time.monotonic()-start
            self.purity_checks+=1
            if ok:self.certified.add(variant)
        if not ok:
            self.invalidate('failed_call')
        elif name not in PURE_QUERIES or self.effects.get(name)!='read_only':
            self.invalidate('non_whitelisted_call')
        elif self.enabled:
            # A query following a morning observation cannot make that observation
            # canonical. Preserve dirty until the NEXT original reload boundary.
            self.successful_queries[name]+=1

    def stats(self):
        return dict(requested_mode=self.mode,enabled=self.enabled,
            fallback_reason=None if self.supported else 'unaudited_core_source',
            reconstructed_boundaries=self.rebuilt,reused_boundaries=self.reused,
            purity_checks=self.purity_checks,audit_seconds=self.audit_seconds,
            successful_whitelisted_queries=dict(self.successful_queries),invalidations=dict(self.invalidations))
