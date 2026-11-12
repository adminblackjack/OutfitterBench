"""Version-gated trip return state and public notices; no private actor oracle."""
from __future__ import annotations

import copy

from ..channel_demand import effective_rental_cash_segment
from ..models import CalendarBlock
from ..return_dynamics import (POLICY_ID, RNG_STREAM, ITINERARIES, LOGISTICS, TYPE_WEIGHTS,
    EXTENSION_ONE_DAY, PUBLIC_RULES, delay_weights, extension_probability, trip_kind)


class ReturnDynamicsMixin:
    def _v46_return_dynamics(self):
        return (int(self.dataset.metadata.get("tool_interface_version", 0)) >= 41
                and self.dataset.metadata.get("return_dynamics_policy") == POLICY_ID)

    def _v46_rng(self, request, name):
        return self._stable_event_rng(RNG_STREAM, *request.event_address, identity=name)

    def _generate_reservation_requests(self, state):
        super()._generate_reservation_requests(state)
        if not self._v46_return_dynamics():
            return
        for request in state.reservation_requests:
            if request.created_day != state.day or request.trip_context:
                continue
            kind = trip_kind(request.lines)
            request.trip_context = {
                "itinerary_type": str(self._v46_rng(request, "itinerary").choice(ITINERARIES, p=TYPE_WEIGHTS[kind])),
                "return_logistics": str(self._v46_rng(request, "logistics").choice(LOGISTICS, p=(.3, .5, .2))),
            }

    def _v46_initialize_trip(self, state, commitment):
        if commitment.return_process_private:
            return
        request = next(row for row in state.reservation_requests if row.request_id == commitment.request_id)
        context = request.trip_context
        due = int(commitment.approved_return_day or commitment.return_day)
        kind = trip_kind(request.lines)
        delay = int(self._v46_rng(request, "independent-return-delay").choice((0, 1, 2),
            p=delay_weights(context["return_logistics"], self.dataset.metadata.get("return_delay_scale", 1.))))
        earliest = min(state.max_days, due + delay)
        extra = 1 if self._v46_rng(request, "extension-length").random() < EXTENSION_ONE_DAY[kind] else 2
        extra = min(extra, state.max_days - due)
        wants_extension = False
        if extra > 0:
            _, _, rent = self._extension_rent_delta(commitment, proposed_return_day=due+extra, bill_fulfilled_only=True)
            primary = next(line for line in request.lines if line.required)
            session = self._restore_quote_context(request.quote_context_by_line[primary.line_id])
            model = self.channel_demand_model
            cash = effective_rental_cash_segment(model, model.segments[request.customer_segment],
                duration_days=due-commitment.pickup_day,
                party_equivalent=max(1., session.adults + .6*session.children),
                request_budget_multiplier=request.request_budget_multiplier)
            probability = extension_probability(kind, context["itinerary_type"], rent, cash.cash_scale_cny)
            wants_extension = bool(self._v46_rng(request, "extension-intent").random() < probability)
        intent_signal = bool(self._v46_rng(request, "extension-signal").random() < (.8 if wants_extension else .2))
        warning = bool(self._v46_rng(request, "return-warning").random() < (.8 if earliest > due else .15))
        commitment.return_process_private = {"pickup_agreed_return_day": due,
            "earliest_feasible_return_day": earliest, "extension_days": extra,
            "wants_extension": wants_extension, "warning": warning,
            "warning_published": False, "eta_published": False, "extension_processed": False,
            "late_counted": False}
        self._v46_notice(state, commitment, "extension_intent_notice", {"may_request_extension": intent_signal})

    def _v46_notice(self, state, commitment, kind, details):
        row = {"day": state.day, "type": kind, **details}
        commitment.return_notices.append(row)
        self._reservation_event(state, kind, request_id=commitment.request_id,
            commitment_id=commitment.commitment_id, details=details)
        state.inbox.append(f"预约 {commitment.commitment_id} 有新的归还/行程通知：{kind}；查看 reservation_inbox。")

    def _v46_effective_return(self, commitment, approved):
        process = commitment.return_process_private
        return (max(int(approved), process["earliest_feasible_return_day"])
                if process.get("eta_published") else int(approved))

    def _v46_sync_return(self, state, commitment):
        approved = int(commitment.approved_return_day or commitment.return_day)
        eta = self._v46_effective_return(commitment, approved)
        previous = int(commitment.estimated_return_day or approved)
        commitment.estimated_return_day = eta
        for contract in state.contracts:
            if contract.reservation_id != commitment.commitment_id or contract.status != "active":
                continue
            extra = eta - contract.return_day
            if extra < 0:
                raise AssertionError("return ETA cannot move backwards in this fixed earliest-return model")
            contract.estimated_return_day = eta
            contract.return_day = eta
            contract.duration_days += extra
            if extra:
                asset = next(row for row in state.assets if row.asset_id == contract.asset_id)
                asset.used_life_days += extra
                self._refresh_contract_lifecycle_damage(asset, contract)
                self._charge_supplier_trial_contract_extension(state, asset, contract,
                    duration_days=extra, reason="供应商试用资产实际迟还使用费")
        for day in range(previous, eta):
            for line in commitment.lines:
                if line.status != "fulfilled":
                    continue
                oid = line.selected_offer_id or line.offer_id
                state.reservation_calendar_blocks.append(CalendarBlock(
                    block_id=f"{commitment.commitment_id}-{line.line_id}-physical-delay-{day}",
                    commitment_id=commitment.commitment_id, line_id=line.line_id, offer_id=oid,
                    day=day, quantity=line.quantity, authorized_offer_ids=[oid], selected_offer_id=oid,
                    block_type="estimated_late_return", promise_profile_id=line.promise_profile_id,
                    profile_schema_version=line.profile_schema_version, service_units=line.service_units,
                    capacity_per_asset=line.capacity_per_asset, staged_offer_id=line.staged_offer_id))

    def _v46_process_trip_notices(self, state):
        if not self._v46_return_dynamics() or state.done:
            return
        for commitment in state.reservation_commitments:
            if commitment.status != "active":
                continue
            self._v46_initialize_trip(state, commitment)
            process = commitment.return_process_private
            due = process["pickup_agreed_return_day"]
            if state.day >= max(commitment.pickup_day, due-2) and not process["warning_published"]:
                process["warning_published"] = True
                self._v46_notice(state, commitment, "return_warning_notice", {"possible_return_delay": process["warning"]})
            if state.day >= due-1 and not process["eta_published"]:
                process["eta_published"] = True
                self._v46_sync_return(state, commitment)
                self._v46_notice(state, commitment, "return_eta_notice", {
                    "approved_return_day": commitment.approved_return_day,
                    "estimated_return_day": commitment.estimated_return_day})
            if state.day >= due-1 and not process["extension_processed"]:
                process["extension_processed"] = True
                if (process["wants_extension"] and self.config.reservation_disruptions_enabled
                        and state.day <= self.config.max_days):
                    self._create_change_request(state, commitment, change_type="extension",
                        proposed_pickup_day=commitment.pickup_day, proposed_return_day=due+process["extension_days"])

    def _process_reservation_pickups(self, state, **kwargs):
        super()._process_reservation_pickups(state, **kwargs)
        self._v46_process_trip_notices(state)

    def _process_morning(self, state):
        # Process notices for trips already underway, including the runoff.
        self._v46_process_trip_notices(state)
        return super()._process_morning(state)

    def _schedule_late_return(self, state, commitment, *, identity):
        if self._v46_return_dynamics():
            # Neither refusal nor timeout draws or changes physical return state.
            return int(commitment.estimated_return_day or commitment.return_day) > int(commitment.approved_return_day or commitment.return_day)
        return super()._schedule_late_return(state, commitment, identity=identity)

    def _update_reservation_after_contract_return(self, state, contract):
        result = super()._update_reservation_after_contract_return(state, contract)
        if self._v46_return_dynamics() and contract.reservation_id:
            commitment = next(row for row in state.reservation_commitments if row.commitment_id == contract.reservation_id)
            process = commitment.return_process_private
            if (process and not process["late_counted"] and contract.actual_return_day
                    and contract.actual_return_day > commitment.approved_return_day):
                process["late_counted"] = True
                state.cumulative_reservation_late_returns += 1
        return result

    @classmethod
    def _public_reservation_request(cls, request):
        result = super()._public_reservation_request(request)
        if not request.trip_context:
            result.pop("trip_context", None)
        return result

    @classmethod
    def _public_reservation_commitment(cls, commitment):
        result = super()._public_reservation_commitment(commitment)
        result.pop("return_process_private", None)
        if not commitment.return_notices:
            result.pop("return_notices", None)
        return result

    @classmethod
    def _v22_reservation_detail_record(cls, request, commitment, **kwargs):
        result = super()._v22_reservation_detail_record(request, commitment, **kwargs)
        if request.trip_context:
            if "current_terms" in result:
                result["current_terms"]["trip_context"] = dict(request.trip_context)
            if commitment and result.get("fulfillment") is not None and "fulfillment" in result:
                result["fulfillment"]["return_notices"] = copy.deepcopy(commitment.return_notices)
        return result

    def reservation_inbox(self, state):
        result = super().reservation_inbox(state)
        if self._v46_return_dynamics():
            result["return_notices"] = [{"commitment_id": c.commitment_id, **copy.deepcopy(notice)}
                for c in state.reservation_commitments for notice in c.return_notices if notice["day"] == state.day]
        return result

    def view_reservation_quote_workspace(self, state, **kwargs):
        result = super().view_reservation_quote_workspace(state, **kwargs)
        if self._v46_return_dynamics():
            contexts = {request.request_id: request.trip_context for request in state.reservation_requests}
            for row in result.get("requests", []):
                row["trip_context"] = dict(contexts[row["request_id"]])
        return result

    def business_manual(self):
        result = super().business_manual()
        if self._v46_return_dynamics():
            result["reservations"]["rules"].append(PUBLIC_RULES)
        return result
