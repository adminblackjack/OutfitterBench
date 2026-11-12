"""V38 delayed customer negotiation for confirmed reservations."""
from __future__ import annotations

import copy
import math
from dataclasses import fields, replace
from typing import Any

from ..contingency_actor_interface import (
    ENVIRONMENT_PROTOCOL as ACTOR_INTERFACE_ENVIRONMENT_PROTOCOL,
    POLICY_ID as ACTOR_INTERFACE_POLICY_ID,
)
from ..errors import ReservationContingencyVersionError, ReservationRecoveryBatchValidationError
from ..models import (
    CalendarBlock,
    ReservationContingencyInquiry,
    ReservationContingencyReport,
    WorldState,
)
from ..reservation_contingency import (
    ENVIRONMENT_PROTOCOL,
    MAX_BATCH_SIZE,
    MAX_PROPOSAL_LEAD_DAYS,
    MAX_PROPOSALS,
    MIN_NOTICE_DAYS,
    POLICY_ID,
    RNG_NAMESPACE,
    acceptance_probability,
    cancellation_probability,
    canonical_proposal,
    counteroffer_probability,
    customer_class,
    date_utility_delta,
    initial_date_is_acceptable,
    normalized_equipment_delta,
    proposal_digest,
)
from ..weather_information import forecast_regime
from ..trip_weather import trip_context, contextualize, FORECAST_DATES


class ReservationContingencyMixin:
    def _v38_proactive_contingency(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 33
            and self.dataset.metadata.get("proactive_contingency_policy")
            == POLICY_ID
        )

    def _v39_contingency_actor_clarity(self) -> bool:
        return bool(
            int(self.dataset.metadata.get("tool_interface_version", 0)) >= 34
            and self.dataset.metadata.get("contingency_actor_interface_policy")
            == ACTOR_INTERFACE_POLICY_ID
        )

    def _validate_reservation_contingency_configuration(self) -> None:
        enabled = bool(self.config.reservation_contingency_enabled)
        if enabled and not self._v38_proactive_contingency():
            raise ValueError(
                "reservation_contingency_enabled requires the V38 runtime"
            )
        if (
            self.dataset.metadata.get("environment_protocol")
            == ENVIRONMENT_PROTOCOL
            and not enabled
        ):
            raise ValueError("V38 requires reservation_contingency_enabled=true")
        if (
            self.dataset.metadata.get("environment_protocol")
            == ACTOR_INTERFACE_ENVIRONMENT_PROTOCOL
            and (not enabled or not self._v39_contingency_actor_clarity())
        ):
            raise ValueError("V39 requires its frozen actor-interface policy")

    def _assert_reservation_contingencies(self, state: WorldState) -> None:
        if not self._v38_proactive_contingency():
            if (
                state.reservation_contingency_inquiries
                or state.reservation_contingency_reports
            ):
                raise ValueError("V38 contingency state cannot load in an older runtime")
            return
        inquiry_ids = [row.inquiry_id for row in state.reservation_contingency_inquiries]
        report_ids = [row.report_id for row in state.reservation_contingency_reports]
        if len(inquiry_ids) != len(set(inquiry_ids)):
            raise ValueError("duplicate contingency inquiry_id")
        if len(report_ids) != len(set(report_ids)):
            raise ValueError("duplicate contingency report_id")
        commitments = {row.commitment_id: row for row in state.reservation_commitments}
        for inquiry in state.reservation_contingency_inquiries:
            if inquiry.commitment_id not in commitments:
                raise ValueError("contingency inquiry references unknown commitment")
            if not 1 <= len(inquiry.proposals) <= MAX_PROPOSALS:
                raise ValueError("invalid persisted contingency proposal count")

    @staticmethod
    def _public_contingency_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
        return {
            "proposal_id": str(proposal["proposal_id"]),
            **canonical_proposal(proposal),
        }

    @classmethod
    def _public_contingency_report(
        cls,
        report: ReservationContingencyReport,
        *,
        actor_clarity: bool = False,
    ) -> dict[str, Any]:
        result = {
            "report_id": report.report_id,
            "inquiry_id": report.inquiry_id,
            "commitment_id": report.commitment_id,
            "evaluated_commitment_version": report.evaluated_commitment_version,
            "response_day": report.response_day,
            "expires_day": report.expires_day,
            "accepted_proposal_ids": list(report.accepted_proposal_ids),
            "rejected_proposal_ids": list(report.rejected_proposal_ids),
            "proposals": [
                cls._public_contingency_proposal(row) for row in report.proposals
            ],
            "counteroffer": (
                None
                if report.counteroffer is None
                else cls._public_contingency_proposal(report.counteroffer)
            ),
            "outcome": report.outcome,
            "status": report.status,
            "available_selection_ids": [
                *report.accepted_proposal_ids,
                *(
                    [str(report.counteroffer["proposal_id"])]
                    if report.counteroffer is not None
                    else []
                ),
            ],
            **({"settlement": copy.deepcopy(report.settlement)} if report.settlement else {}),
        }
        if not actor_clarity or report.status == "active_actionable":
            result["application_deadline"] = "end_of_response_business_day"
        return result

    def _contingency_customer_context(self, request: Any) -> dict[str, Any]:
        required = next((line for line in request.lines if line.required), request.lines[0])
        session = self._restore_quote_context(
            request.quote_context_by_line[required.line_id]
        )
        return {
            "customer_type": customer_class(
                children=session.children, archetype=session.archetype
            ),
            "children": int(session.children),
            "archetype": str(session.archetype),
            "schedule_flexibility": float(request.schedule_flexibility),
            "forecast_trust": float(request.forecast_trust),
            "occasion_importance": float(session.occasion_importance),
        }

    def _contingency_forecast_snapshot(
        self, state: WorldState, *, through_day: int
    ) -> dict[str, dict[str, Any]]:
        horizon = max(1, int(through_day) - state.day + 1)
        if self._v44_trip_weather():
            horizon = FORECAST_DATES
            through_day = min(state.max_days, state.day + FORECAST_DATES - 1)
        forecasts = {
            int(row["day"]): row for row in self.weather_forecast(state, horizon)
        }
        return {
            str(day): {
                "day": day,
                "condition": str(forecasts[day]["condition"]),
                "condition_accuracy": float(forecasts[day]["condition_accuracy"]),
                "temp_mean_c": float(forecasts[day]["temp_mean_c"]),
                "night_low_c": float(forecasts[day]["night_low_c"]),
                "day_high_c": float(forecasts[day]["day_high_c"]),
                "day_type": str(self._calendar(day)["day_type"]),
                **({'is_school_holiday': self._calendar(day)['is_school_holiday']}
                   if self._v49_school_holiday() else {}),
            }
            for day in range(state.day, int(through_day) + 1)
        }

    def _contingency_date_delta(
        self,
        *,
        commitment: Any,
        request: Any,
        candidate_pickup_day: int,
        snapshot: dict[str, dict[str, Any]],
    ) -> float:
        context = self._contingency_customer_context(request)
        current = snapshot[str(commitment.pickup_day)]
        candidate = snapshot[str(candidate_pickup_day)]
        return date_utility_delta(
            customer_type=context["customer_type"],
            current_day_type=str(current["day_type"]),
            candidate_day_type=str(candidate["day_type"]),
            current_is_school_holiday=bool(current.get('is_school_holiday', False)),
            candidate_is_school_holiday=bool(candidate.get('is_school_holiday', False)),
            current_condition=(
                forecast_regime(current)
                if self._v42_weather_information_coherence()
                else str(current["condition"])
            ),
            candidate_condition=(
                forecast_regime(candidate)
                if self._v42_weather_information_coherence()
                else str(candidate["condition"])
            ),
            shift_days=candidate_pickup_day - commitment.pickup_day,
            schedule_flexibility=context["schedule_flexibility"],
            forecast_trust=context["forecast_trust"],
            occasion_importance=context["occasion_importance"],
        )

    def _contingency_equipment_raw_delta(
        self,
        state: WorldState,
        request: Any,
        commitment: Any,
        line_changes: list[dict[str, Any]],
        *,
        pickup_day: int,
        forecast_snapshot: dict[str, dict[str, Any]],
        evaluation_day: int,
    ) -> float:
        """Compare frozen product fit while holding all money terms constant."""
        forecast = forecast_snapshot[str(pickup_day)]
        dated_request = replace(
            request,
            quote_context_by_line={
                line_id: {
                    **copy.deepcopy(context),
                    "pickup_day": int(pickup_day),
                    "booking_lead_days": max(0, int(pickup_day) - int(evaluation_day)),
                    "forecast_adversity": self._forecast_adversity(
                        forecast_regime(forecast)
                        if self._v42_weather_information_coherence()
                        else str(forecast["condition"])
                    ),
                    "forecast_uncertainty": max(
                        0.0, 1.0 - float(forecast["condition_accuracy"])
                    ),
                }
                for line_id, context in request.quote_context_by_line.items()
            },
            quote_context_by_date={},
        )
        if self._v44_trip_weather():
            context = trip_context(
                issued_day=evaluation_day, pickup_day=pickup_day,
                duration_days=commitment.return_day - commitment.pickup_day,
                forecast_rows=list(forecast_snapshot.values()))
            dated_request = replace(dated_request, quote_context_by_line={
                lid: contextualize(raw, context)
                for lid, raw in dated_request.quote_context_by_line.items()})
        changes = {str(row["line_id"]): row for row in line_changes}
        total = 0.0
        for line in commitment.lines:
            change = changes.get(line.line_id)
            if change is None:
                continue
            current_offer = line.selected_offer_id or line.offer_id
            proposed_offer = str(change["offer_id"])
            if proposed_offer == current_offer:
                continue
            quantity = int(line.asset_quantity or line.quantity)
            unit_price = (
                float(line.locked_line_daily_rent) / max(1, quantity)
                if line.locked_line_daily_rent is not None
                else float(line.locked_daily_price)
            )
            common = {
                "promise_profile_id": str(line.promise_profile_id),
                "asset_quantity": quantity,
                "daily_unit_price": unit_price,
                "total_rent": float(line.locked_rent),
                "locked_security_deposit": float(line.locked_security_deposit),
            }
            current = self._bundle_store_option(
                state,
                dated_request,
                line,
                {"option_id": "current", "offer_id": current_offer, **common},
            )
            proposed = self._bundle_store_option(
                state,
                dated_request,
                line,
                {"option_id": "proposed", "offer_id": proposed_offer, **common},
            )
            if current is None or proposed is None:
                raise AssertionError("same-Profile contingency fit could not be evaluated")
            total += float(proposed.non_cash_value) - float(current.non_cash_value)
        return total

    def _normalize_contingency_proposal(
        self,
        state: WorldState,
        request: Any,
        commitment: Any,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = {"proposal_id", "pickup_day", "return_day", "line_changes"}
        unexpected = sorted(set(raw) - allowed)
        if unexpected:
            raise ValueError(f"unexpected proposal fields: {unexpected}")
        proposal_id = str(raw.get("proposal_id", ""))
        if not proposal_id or len(proposal_id) > 60:
            raise ValueError("proposal_id must contain 1-60 characters")
        pickup = raw.get("pickup_day")
        returned = raw.get("return_day")
        if type(pickup) is not int or type(returned) is not int:
            raise ValueError("proposal pickup_day and return_day must be integers")
        duration = int(commitment.return_day) - int(commitment.pickup_day)
        if returned - pickup != duration:
            raise ValueError("contingency proposals must preserve rental duration")
        if not MIN_NOTICE_DAYS <= pickup - state.day <= MAX_PROPOSAL_LEAD_DAYS:
            raise ValueError("proposal pickup_day must be 2-14 days ahead")
        if pickup > self.config.max_days or returned > state.max_days:
            raise ValueError("proposal dates exceed the reservation horizon")
        line_changes = raw.get("line_changes", [])
        if not isinstance(line_changes, list) or any(
            not isinstance(row, dict) for row in line_changes
        ):
            raise ValueError("line_changes must be an array of objects")
        by_line = {line.line_id: line for line in commitment.lines}
        normalized_changes = []
        seen = set()
        for row in line_changes:
            if set(row) != {"line_id", "offer_id", "promise_profile_id"}:
                raise ValueError(
                    "each line change requires line_id, offer_id, and promise_profile_id"
                )
            line_id = str(row["line_id"])
            if line_id in seen or line_id not in by_line:
                raise ValueError("line_changes must reference unique commitment lines")
            seen.add(line_id)
            line = by_line[line_id]
            profile_id = str(row["promise_profile_id"])
            offer_id = str(row["offer_id"])
            if profile_id != line.promise_profile_id:
                raise ValueError("equipment changes must retain the current Promise Profile")
            legal_offer_ids = self._contingency_same_profile_offer_ids(
                request, line
            )
            if offer_id not in legal_offer_ids:
                raise ValueError("offer_id is not a frozen member of the current Profile")
            normalized_changes.append(
                {
                    "line_id": line_id,
                    "offer_id": offer_id,
                    "promise_profile_id": profile_id,
                }
            )
        proposal = {
            "proposal_id": proposal_id,
            "pickup_day": pickup,
            "return_day": returned,
            "line_changes": normalized_changes,
        }
        if pickup == commitment.pickup_day and not normalized_changes:
            raise ValueError("proposal must materially change date or equipment")
        return proposal

    def request_reservation_contingency(
        self, state: WorldState, inquiries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not self._v38_proactive_contingency():
            raise ValueError("proactive contingency requires V38")
        if not inquiries or len(inquiries) > MAX_BATCH_SIZE:
            raise ValueError("inquiries must contain 1-25 items")
        commitments = {row.commitment_id: row for row in state.reservation_commitments}
        requests = {row.request_id: row for row in state.reservation_requests}
        checked = []
        identifiers = []
        for row in inquiries:
            if set(row) != {"commitment_id", "commitment_version", "proposals"}:
                raise ValueError(
                    "each inquiry requires commitment_id, commitment_version, and proposals"
                )
            commitment_id = str(row["commitment_id"])
            identifiers.append(commitment_id)
            commitment = commitments.get(commitment_id)
            if commitment is None:
                raise ValueError(f"unknown commitment_id: {commitment_id}")
            if commitment.status != "confirmed" or commitment.assignments:
                raise ValueError("contingency inquiry requires a confirmed pre-pickup contract")
            if state.day > commitment.pickup_day - MIN_NOTICE_DAYS:
                raise ValueError("contingency inquiry must be submitted by pickup_day - 2")
            if type(row["commitment_version"]) is not int or int(
                row["commitment_version"]
            ) != int(commitment.commitment_version):
                if self._v39_contingency_actor_clarity():
                    raise ReservationContingencyVersionError(
                        commitment_id=commitment_id,
                        received_version=row["commitment_version"],
                        current_version=commitment.commitment_version,
                    )
                raise ValueError("commitment_version is stale or invalid")
            if commitment.contingency_inquiry_used:
                raise ValueError("this commitment has already used its contingency inquiry")
            proposals = row["proposals"]
            if not isinstance(proposals, list) or not 1 <= len(proposals) <= MAX_PROPOSALS:
                raise ValueError("each inquiry needs one or two proposals")
            request = requests[commitment.request_id]
            normalized = [
                self._normalize_contingency_proposal(
                    state, request, commitment, dict(proposal)
                )
                for proposal in proposals
            ]
            proposal_ids = [proposal["proposal_id"] for proposal in normalized]
            digests = [proposal_digest(proposal) for proposal in normalized]
            if len(set(proposal_ids)) != len(proposal_ids):
                raise ValueError("proposal_id values must be unique within an inquiry")
            if len(set(digests)) != len(digests):
                raise ValueError("duplicate semantic proposals are not allowed")
            snapshot = self._contingency_forecast_snapshot(
                state,
                through_day=max(
                    commitment.pickup_day,
                    *(proposal["pickup_day"] for proposal in normalized),
                    min(state.max_days, state.day + MAX_PROPOSAL_LEAD_DAYS),
                ),
            )
            checked.append((commitment, normalized, snapshot))
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("a batch may contain each commitment only once")

        results = []
        for commitment, proposals, snapshot in checked:
            inquiry_id = (
                f"contingency-inquiry-{state.next_reservation_contingency_inquiry_id:06d}"
            )
            state.next_reservation_contingency_inquiry_id += 1
            inquiry = ReservationContingencyInquiry(
                inquiry_id=inquiry_id,
                commitment_id=commitment.commitment_id,
                evaluated_commitment_version=commitment.commitment_version,
                submitted_day=state.day,
                response_day=state.day + 1,
                proposals=copy.deepcopy(proposals),
                forecast_snapshot_by_day=copy.deepcopy(snapshot),
            )
            state.reservation_contingency_inquiries.append(inquiry)
            commitment.contingency_inquiry_used = True
            state.cumulative_reservation_contingency_inquiries += 1
            self._reservation_event(
                state,
                "reservation_contingency_requested",
                request_id=commitment.request_id,
                commitment_id=commitment.commitment_id,
                details={
                    "inquiry_id": inquiry_id,
                    "evaluated_commitment_version": commitment.commitment_version,
                    "response_day": state.day + 1,
                    "proposal_count": len(proposals),
                },
            )
            results.append(
                {
                    "inquiry_id": inquiry_id,
                    "commitment_id": commitment.commitment_id,
                    "evaluated_commitment_version": commitment.commitment_version,
                    "response_day": state.day + 1,
                    "proposal_ids": [row["proposal_id"] for row in proposals],
                }
            )
        self._assert_reservation_contingencies(state)
        return {"inquiries": results}

    def _counteroffer_candidate(
        self,
        state: WorldState,
        commitment: Any,
        request: Any,
        inquiry: ReservationContingencyInquiry,
    ) -> dict[str, Any] | None:
        duration = commitment.return_day - commitment.pickup_day
        proposed_days = {int(row["pickup_day"]) for row in inquiry.proposals}
        candidates = []
        for day in range(
            inquiry.submitted_day + MIN_NOTICE_DAYS,
            inquiry.submitted_day + MAX_PROPOSAL_LEAD_DAYS + 1,
        ):
            if (
                day == commitment.pickup_day
                or day in proposed_days
                or day > self.config.max_days
                or day + duration > state.max_days
            ):
                continue
            delta = self._contingency_date_delta(
                commitment=commitment,
                request=request,
                candidate_pickup_day=day,
                snapshot=inquiry.forecast_snapshot_by_day,
            )
            if initial_date_is_acceptable(delta, request.schedule_flexibility):
                candidates.append((day, delta))
        if not candidates:
            return None
        candidates.sort(key=lambda row: row[0])
        maximum = max(score for _, score in candidates)
        temperature = float(
            self.dataset.metadata["proactive_contingency"]["parameters"]
            ["initial_sampling_temperature"]
        )
        weights = [math.exp(temperature * (score - maximum)) for _, score in candidates]
        rng = self._stable_event_rng(
            RNG_NAMESPACE,
            *request.event_address,
            identity=(
                f"contingency-counteroffer:v{inquiry.evaluated_commitment_version}:"
                + ":".join(sorted(proposal_digest(row) for row in inquiry.proposals))
            ),
        )
        draw = float(rng.random()) * sum(weights)
        cumulative = 0.0
        day = candidates[-1][0]
        for (candidate_day, _), weight in zip(candidates, weights):
            cumulative += weight
            if draw <= cumulative:
                day = candidate_day
                break
        return {
            "proposal_id": "customer-counteroffer",
            "pickup_day": day,
            "return_day": day + duration,
            "line_changes": [],
        }

    def _process_contingency_responses(self, state: WorldState) -> None:
        if not self._v38_proactive_contingency():
            return
        commitments = {row.commitment_id: row for row in state.reservation_commitments}
        requests = {row.request_id: row for row in state.reservation_requests}
        for inquiry in sorted(
            state.reservation_contingency_inquiries, key=lambda row: row.inquiry_id
        ):
            if inquiry.status != "pending" or inquiry.response_day != state.day:
                continue
            commitment = commitments[inquiry.commitment_id]
            request = requests[commitment.request_id]
            if (
                commitment.status != "confirmed"
                or commitment.commitment_version
                != inquiry.evaluated_commitment_version
            ):
                inquiry.status = "stale"
                if self._v39_contingency_actor_clarity():
                    report_id = (
                        "contingency-report-"
                        f"{state.next_reservation_contingency_report_id:06d}"
                    )
                    state.next_reservation_contingency_report_id += 1
                    report = ReservationContingencyReport(
                        report_id=report_id,
                        inquiry_id=inquiry.inquiry_id,
                        commitment_id=commitment.commitment_id,
                        evaluated_commitment_version=(
                            inquiry.evaluated_commitment_version
                        ),
                        response_day=state.day,
                        expires_day=state.day,
                        proposals=copy.deepcopy(inquiry.proposals),
                        outcome="stale_commitment_version",
                        status="stale",
                    )
                    state.reservation_contingency_reports.append(report)
                    inquiry.report_id = report_id
                    state.inbox.append(
                        f"预约 {commitment.commitment_id} 的主动协商询问已失效；"
                        f"结构化更新见报告 {report_id}。"
                    )
                self._reservation_event(
                    state,
                    "reservation_contingency_stale",
                    request_id=request.request_id,
                    commitment_id=commitment.commitment_id,
                    details={"inquiry_id": inquiry.inquiry_id},
                )
                continue
            context = self._contingency_customer_context(request)
            accepted = []
            utilities = {}
            for proposal in inquiry.proposals:
                utility = self._contingency_date_delta(
                    commitment=commitment,
                    request=request,
                    candidate_pickup_day=int(proposal["pickup_day"]),
                    snapshot=inquiry.forecast_snapshot_by_day,
                ) + normalized_equipment_delta(
                    self._contingency_equipment_raw_delta(
                        state,
                        request,
                        commitment,
                        list(proposal.get("line_changes", [])),
                        pickup_day=int(proposal["pickup_day"]),
                        forecast_snapshot=inquiry.forecast_snapshot_by_day,
                        evaluation_day=inquiry.submitted_day,
                    )
                )
                utilities[proposal["proposal_id"]] = utility
                probability = acceptance_probability(
                    utility_delta=utility,
                    schedule_flexibility=context["schedule_flexibility"],
                    notice_days=commitment.pickup_day - inquiry.submitted_day,
                )
                rng = self._stable_event_rng(
                    RNG_NAMESPACE,
                    *request.event_address,
                    identity=(
                        f"contingency-accept:v{inquiry.evaluated_commitment_version}:"
                        f"{proposal_digest(proposal)}"
                    ),
                )
                if rng.random() < probability:
                    accepted.append(str(proposal["proposal_id"]))
            rejected = [
                str(row["proposal_id"])
                for row in inquiry.proposals
                if str(row["proposal_id"]) not in accepted
            ]
            counteroffer = None
            outcome = "proposal_accepted" if accepted else "keep_original"
            settlement: dict[str, Any] = {}
            if not accepted:
                counter_candidate = self._counteroffer_candidate(
                    state, commitment, request, inquiry
                )
                best = max(
                    inquiry.proposals,
                    key=lambda row: utilities[str(row["proposal_id"])],
                )
                cancel_p = cancellation_probability(
                    best_utility_delta=utilities[str(best["proposal_id"])],
                    best_shift_days=int(best["pickup_day"]) - commitment.pickup_day,
                    schedule_flexibility=context["schedule_flexibility"],
                    notice_days=commitment.pickup_day - inquiry.submitted_day,
                )
                counter_p = counteroffer_probability(
                    candidate_available=counter_candidate is not None,
                    schedule_flexibility=context["schedule_flexibility"],
                    notice_days=commitment.pickup_day - inquiry.submitted_day,
                )
                canonical_set = sorted(proposal_digest(row) for row in inquiry.proposals)
                residual_rng = self._stable_event_rng(
                    RNG_NAMESPACE,
                    *request.event_address,
                    identity=(
                        f"contingency-residual:v{inquiry.evaluated_commitment_version}:"
                        + ":".join(canonical_set)
                    ),
                )
                draw = float(residual_rng.random())
                if draw < cancel_p:
                    settlement = self._fail_reservation_commitment(
                        state,
                        commitment,
                        status="merchant_induced_customer_cancellation",
                        cause="proactive_contingency_customer_cancellation",
                        notice_days=max(0, commitment.pickup_day - state.day),
                    )
                    state.cumulative_reservation_contingency_customer_cancellations += 1
                    outcome = "merchant_induced_customer_cancellation"
                elif draw < cancel_p + counter_p and counter_candidate is not None:
                    counteroffer = counter_candidate
                    outcome = "counteroffer"
            report_id = (
                f"contingency-report-{state.next_reservation_contingency_report_id:06d}"
            )
            state.next_reservation_contingency_report_id += 1
            report = ReservationContingencyReport(
                report_id=report_id,
                inquiry_id=inquiry.inquiry_id,
                commitment_id=commitment.commitment_id,
                evaluated_commitment_version=inquiry.evaluated_commitment_version,
                response_day=state.day,
                expires_day=state.day,
                proposals=copy.deepcopy(inquiry.proposals),
                accepted_proposal_ids=accepted,
                rejected_proposal_ids=rejected,
                counteroffer=copy.deepcopy(counteroffer),
                outcome=outcome,
                status=(
                    (
                        "terminal_customer_cancellation"
                        if outcome == "merchant_induced_customer_cancellation"
                        else "active_actionable"
                        if accepted or counteroffer is not None
                        else "terminal_keep_original"
                    )
                    if self._v39_contingency_actor_clarity()
                    else "terminal_customer_cancellation"
                    if outcome == "merchant_induced_customer_cancellation"
                    else "active"
                ),
                settlement=settlement,
            )
            state.reservation_contingency_reports.append(report)
            inquiry.status = "reported"
            inquiry.report_id = report_id
            if report.status in {"active", "active_actionable"}:
                state.inbox.append(
                    f"预约 {commitment.commitment_id} 的主动协商报告 {report_id} 已发布；"
                    "如需变更，须在今日关班前明确应用。"
                )
            elif report.status == "terminal_customer_cancellation":
                state.inbox.append(
                    f"预约 {commitment.commitment_id} 在主动协商后由客户取消；"
                    f"结算记录见报告 {report_id}。"
                )
            elif report.status == "terminal_keep_original":
                state.inbox.append(
                    f"预约 {commitment.commitment_id} 的主动协商已结束；"
                    f"客户保留原合同，结构化更新见报告 {report_id}。"
                )
            self._reservation_event(
                state,
                "reservation_contingency_reported",
                request_id=request.request_id,
                commitment_id=commitment.commitment_id,
                details={
                    "inquiry_id": inquiry.inquiry_id,
                    "report_id": report_id,
                    "outcome": outcome,
                    "accepted_proposal_ids": accepted,
                    "counteroffer_available": counteroffer is not None,
                    **({"settlement": settlement} if settlement else {}),
                },
            )

    def _rebuild_version_recovery(
        self,
        state: WorldState,
        commitment: Any,
        *,
        forecast_snapshot: dict[str, dict[str, Any]] | None = None,
        evaluation_day: int | None = None,
    ) -> None:
        request = next(
            row for row in state.reservation_requests if row.request_id == commitment.request_id
        )
        if forecast_snapshot is None:
            if commitment.pickup_day < state.day:
                forecast_snapshot = {
                    str(commitment.pickup_day): {
                        "condition": request.forecast_condition,
                        "condition_accuracy": request.forecast_accuracy,
                        "day_type": str(self._calendar(commitment.pickup_day)["day_type"]),
                        **({'is_school_holiday': self._calendar(commitment.pickup_day)['is_school_holiday']}
                           if self._v49_school_holiday() else {}),
                    }
                }
            else:
                forecast_snapshot = self._contingency_forecast_snapshot(
                    state, through_day=commitment.pickup_day
                )
        if evaluation_day is None:
            evaluation_day = state.day
        rebuilt = {}
        for line in commitment.lines:
            profile_id = str(line.promise_profile_id or "")
            members = request.inquiry_profile_member_ids_by_line.get(
                line.line_id, {}
            ).get(profile_id, [])
            current = line.selected_offer_id or line.offer_id
            tolerance = 0.15 + 0.50 * float(request.schedule_flexibility)
            authorized = []
            for offer_id in members:
                if offer_id == current:
                    continue
                raw_delta = self._contingency_equipment_raw_delta(
                    state,
                    request,
                    commitment,
                    [
                        {
                            "line_id": line.line_id,
                            "offer_id": offer_id,
                            "promise_profile_id": profile_id,
                        }
                    ],
                    pickup_day=commitment.pickup_day,
                    forecast_snapshot=forecast_snapshot,
                    evaluation_day=evaluation_day,
                )
                if normalized_equipment_delta(raw_delta) >= -tolerance:
                    authorized.append(offer_id)
            rebuilt[line.line_id] = sorted(authorized)
        commitment.version_recovery_offer_ids_by_line = rebuilt
        commitment.preauthorized_recovery_plans = []
        commitment.selected_recovery_plan_id = None
        commitment.recovery_path = None
        commitment.fallback_inquiry_day = None
        commitment.fallback_revealed_offer_ids_by_line = {}

    def _increment_commitment_version(
        self,
        state: WorldState,
        commitment: Any,
        *,
        rebuild_recovery: bool = True,
        forecast_snapshot: dict[str, dict[str, Any]] | None = None,
        evaluation_day: int | None = None,
    ) -> None:
        if not self._v38_proactive_contingency():
            return
        commitment.commitment_version += 1
        if self._v45_reservation_difficulty():
            commitment.partial_fulfillment_inquiry_day = None
            commitment.partial_fulfillment_missing_line_ids = []
            commitment.partial_fulfillment_accepted = None
            commitment.partial_fulfillment_selected = False
            commitment.partial_fulfillment_compensation = 0.0
        if rebuild_recovery:
            self._rebuild_version_recovery(
                state,
                commitment,
                forecast_snapshot=forecast_snapshot,
                evaluation_day=evaluation_day,
            )
        if (
            self._v42_weather_information_coherence()
            and commitment.status == "confirmed"
            and commitment.pickup_day > state.day
        ):
            self._reset_v42_weather_review(
                state,
                commitment,
                forecast_snapshot=forecast_snapshot,
            )
        for change in state.reservation_change_requests:
            if (
                change.commitment_id == commitment.commitment_id
                and change.status == "pending"
                and change.evaluated_commitment_version is not None
                and change.evaluated_commitment_version != commitment.commitment_version
            ):
                change.status = "stale_commitment_version"
                change.decision_day = state.day

    def _create_change_request(self, state: WorldState, commitment: Any, **kwargs: Any):
        before = len(state.reservation_change_requests)
        result = super()._create_change_request(state, commitment, **kwargs)
        if self._v38_proactive_contingency():
            for change in state.reservation_change_requests[before:]:
                change.evaluated_commitment_version = commitment.commitment_version
                if self._v44_trip_weather() and change.change_type == "reschedule":
                    change.weather_forecast_snapshot_by_day = self._contingency_forecast_snapshot(
                        state, through_day=change.proposed_pickup_day)
        return result

    def _apply_reservation_change(self, state: WorldState, change: Any):
        commitment = next(
            row
            for row in state.reservation_commitments
            if row.commitment_id == change.commitment_id
        )
        if (
            self._v38_proactive_contingency()
            and change.evaluated_commitment_version is not None
            and change.evaluated_commitment_version != commitment.commitment_version
        ):
            raise ValueError("reservation change is stale for current contract terms")
        result = super()._apply_reservation_change(state, change)
        self._increment_commitment_version(
            state,
            commitment,
            rebuild_recovery=change.change_type == "reschedule",
            **({"forecast_snapshot": change.weather_forecast_snapshot_by_day,
                "evaluation_day": change.created_day}
               if self._v44_trip_weather() and change.change_type == "reschedule" else {}),
        )
        return result

    def respond_reservation_changes(self, state: WorldState, decisions: list[dict[str, Any]]):
        if self._v38_proactive_contingency():
            changes = {row.change_id: row for row in state.reservation_change_requests}
            commitments = {row.commitment_id: row for row in state.reservation_commitments}
            for decision in decisions:
                change = changes.get(str(decision.get("change_id", "")))
                if (
                    change is not None
                    and change.evaluated_commitment_version is not None
                    and change.evaluated_commitment_version
                    != commitments[change.commitment_id].commitment_version
                ):
                    raise ValueError("reservation change is stale for current contract terms")
        return super().respond_reservation_changes(state, decisions)

    def resolve_reservation_risks(self, state: WorldState, actions: list[dict[str, Any]]):
        if self._v51_recovery_integrity() and len(actions) != 1 and any(
            row.get('action') in {'inquire_fallback', 'inquire_partial_fulfillment'}
            for row in actions
        ):
            # Reject before contacting anyone. A real customer response must
            # never be disclosed and then undone by a later batch error.
            raise ReservationRecoveryBatchValidationError(
                'customer inquiry must be the only action in this call; '
                'read the response before submitting selection actions')
        if int(self.dataset.metadata.get("tool_interface_version", 0)) < 44:
            return self._resolve_reservation_risks_with_versions(state, actions)
        snapshot = copy.deepcopy(state)
        try:
            result = self._resolve_reservation_risks_with_versions(state, actions)
        except Exception as error:
            # WorldState is a slotted dataclass. Restore every field, including
            # ledgers, counters, inquiry rights and event IDs, on any exception.
            # Preserve references for rejected no-op validations (for example a
            # repeated inquiry). After a real rollback, state is authoritative.
            if state != snapshot:
                for field in fields(state):
                    setattr(state, field.name, getattr(snapshot, field.name))
            if isinstance(error, (KeyError, TypeError, ValueError)):
                raise ReservationRecoveryBatchValidationError(str(error)) from error
            raise
        result["batch_atomic"] = True
        return result

    def _resolve_reservation_risks_with_versions(self, state, actions):
        before = {
            row.commitment_id: (
                row.commitment_version,
                tuple(
                    (line.line_id, line.selected_offer_id or line.offer_id)
                    for line in row.lines
                ),
                row.selected_recovery_plan_id,
            )
            for row in state.reservation_commitments
        }
        result = super().resolve_reservation_risks(state, actions)
        if self._v38_proactive_contingency():
            by_id = {row.commitment_id: row for row in state.reservation_commitments}
            updated = set()
            for action in actions:
                if action.get("action") not in {
                    "select_preauthorized_plan",
                    "select_authorized_fallback",
                }:
                    continue
                commitment_id = str(action["commitment_id"])
                if int(self.dataset.metadata.get("tool_interface_version", 0)) >= 44:
                    if commitment_id in updated or commitment_id not in before:
                        continue
                    updated.add(commitment_id)
                commitment = by_id[commitment_id]
                previous = before[commitment.commitment_id]
                current_signature = tuple(
                    (line.line_id, line.selected_offer_id or line.offer_id)
                    for line in commitment.lines
                )
                if (
                    current_signature != previous[1]
                    or commitment.selected_recovery_plan_id != previous[2]
                ):
                    self._increment_commitment_version(
                        state, commitment, rebuild_recovery=False
                    )
        return result

    def _apply_contingency_terms(
        self,
        state: WorldState,
        commitment: Any,
        proposal: dict[str, Any],
        *,
        forecast_snapshot: dict[str, dict[str, Any]],
        evaluation_day: int,
    ) -> None:
        for block in state.reservation_calendar_blocks:
            if block.commitment_id == commitment.commitment_id and block.status in {
                "active",
                "bound",
            }:
                block.status = "contingency_superseded"
        by_line = {line.line_id: line for line in commitment.lines}
        for change in proposal.get("line_changes", []):
            line = by_line[str(change["line_id"])]
            line.offer_id = str(change["offer_id"])
            line.selected_offer_id = str(change["offer_id"])
            line.authorized_offer_ids = [str(change["offer_id"])]
            line.staged_offer_id = None
        commitment.pickup_day = int(proposal["pickup_day"])
        commitment.return_day = int(proposal["return_day"])
        commitment.promised_return_day = commitment.return_day
        commitment.approved_return_day = commitment.return_day
        commitment.estimated_return_day = commitment.return_day
        for line in commitment.lines:
            selected = line.selected_offer_id or line.offer_id
            for day in range(commitment.pickup_day, commitment.return_day):
                state.reservation_calendar_blocks.append(
                    CalendarBlock(
                        block_id=(
                            f"{commitment.commitment_id}-{line.line_id}-"
                            f"contingency-v{commitment.commitment_version + 1}-{day:03d}"
                        ),
                        commitment_id=commitment.commitment_id,
                        line_id=line.line_id,
                        offer_id=selected,
                        day=day,
                        quantity=line.quantity,
                        authorized_offer_ids=[selected],
                        selected_offer_id=selected,
                        block_type="contingency_commitment",
                        promise_profile_id=line.promise_profile_id,
                        profile_schema_version=line.profile_schema_version,
                        service_units=line.service_units,
                        capacity_per_asset=line.capacity_per_asset,
                    )
                )
        self._increment_commitment_version(
            state,
            commitment,
            forecast_snapshot=forecast_snapshot,
            evaluation_day=evaluation_day,
        )

    def apply_reservation_contingency(
        self, state: WorldState, decisions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not self._v38_proactive_contingency():
            raise ValueError("proactive contingency requires V38")
        if not decisions or len(decisions) > MAX_BATCH_SIZE:
            raise ValueError("decisions must contain 1-25 items")
        reports = {row.report_id: row for row in state.reservation_contingency_reports}
        commitments = {row.commitment_id: row for row in state.reservation_commitments}
        checked = []
        ids = []
        for row in decisions:
            if set(row) != {"report_id", "selection_id"}:
                raise ValueError("each decision requires report_id and selection_id")
            report_id = str(row["report_id"])
            ids.append(report_id)
            report = reports.get(report_id)
            if report is None:
                raise ValueError(f"unknown report_id: {report_id}")
            valid_status = (
                "active_actionable"
                if self._v39_contingency_actor_clarity()
                else "active"
            )
            if report.status != valid_status or report.response_day != state.day:
                raise ValueError("contingency report is not active today")
            commitment = commitments[report.commitment_id]
            if (
                commitment.status != "confirmed"
                or commitment.commitment_version
                != report.evaluated_commitment_version
            ):
                raise ValueError("contingency report is stale for current contract terms")
            selection_id = str(row["selection_id"])
            choices = {
                str(proposal["proposal_id"]): proposal
                for proposal in report.proposals
                if str(proposal["proposal_id"]) in report.accepted_proposal_ids
            }
            if report.counteroffer is not None:
                choices[str(report.counteroffer["proposal_id"])] = report.counteroffer
            if selection_id not in choices:
                raise ValueError("selection_id was not accepted or counteroffered")
            checked.append((report, commitment, selection_id, choices[selection_id]))
        if len(ids) != len(set(ids)):
            raise ValueError("a batch may contain each report only once")
        results = []
        for report, commitment, selection_id, proposal in checked:
            inquiry = next(
                row
                for row in state.reservation_contingency_inquiries
                if row.inquiry_id == report.inquiry_id
            )
            previous = {
                "pickup_day": commitment.pickup_day,
                "return_day": commitment.return_day,
                "commitment_version": commitment.commitment_version,
                "line_offer_ids": {
                    line.line_id: line.selected_offer_id or line.offer_id
                    for line in commitment.lines
                },
            }
            self._apply_contingency_terms(
                state,
                commitment,
                proposal,
                forecast_snapshot=inquiry.forecast_snapshot_by_day,
                evaluation_day=inquiry.submitted_day,
            )
            report.status = "applied"
            state.cumulative_reservation_contingencies_applied += 1
            self._reservation_event(
                state,
                "reservation_contingency_applied",
                request_id=commitment.request_id,
                commitment_id=commitment.commitment_id,
                details={
                    "report_id": report.report_id,
                    "selection_id": selection_id,
                    "previous_terms": previous,
                    "new_pickup_day": commitment.pickup_day,
                    "new_return_day": commitment.return_day,
                    "commitment_version": commitment.commitment_version,
                },
            )
            results.append(
                {
                    "report_id": report.report_id,
                    "commitment_id": commitment.commitment_id,
                    "selection_id": selection_id,
                    "commitment_version": commitment.commitment_version,
                    "pickup_day": commitment.pickup_day,
                    "return_day": commitment.return_day,
                    "locked_financial_terms_unchanged": True,
                }
            )
        return {"applications": results, "cash": round(state.cash, 2)}

    def _expire_contingency_reports(self, state: WorldState) -> None:
        if not self._v38_proactive_contingency():
            return
        expirable_status = (
            "active_actionable"
            if self._v39_contingency_actor_clarity()
            else "active"
        )
        for report in state.reservation_contingency_reports:
            if report.status == expirable_status and report.expires_day == state.day:
                report.status = "expired_unapplied"
                commitment = next(
                    row
                    for row in state.reservation_commitments
                    if row.commitment_id == report.commitment_id
                )
                if self._v39_contingency_actor_clarity():
                    self._reservation_event(
                        state,
                        "reservation_contingency_expired",
                        request_id=commitment.request_id,
                        commitment_id=commitment.commitment_id,
                        details={"report_id": report.report_id},
                    )

    def _refresh_contingency_staleness(self, state: WorldState) -> None:
        if not self._v38_proactive_contingency():
            return
        commitments = {row.commitment_id: row for row in state.reservation_commitments}
        staleable_status = (
            "active_actionable"
            if self._v39_contingency_actor_clarity()
            else "active"
        )
        for report in state.reservation_contingency_reports:
            if report.status != staleable_status:
                continue
            commitment = commitments[report.commitment_id]
            if (
                commitment.status != "confirmed"
                or commitment.commitment_version
                != report.evaluated_commitment_version
            ):
                report.status = "stale"
                self._reservation_event(
                    state,
                    (
                        "reservation_contingency_stale"
                        if self._v39_contingency_actor_clarity()
                        else "reservation_contingency_expired"
                    ),
                    request_id=commitment.request_id,
                    commitment_id=commitment.commitment_id,
                    details={"report_id": report.report_id},
                )

    def _process_morning(self, state: WorldState) -> None:
        super()._process_morning(state)
        self._process_contingency_responses(state)

    def _expire_reservation_requests(self, state: WorldState) -> None:
        self._expire_contingency_reports(state)
        super()._expire_reservation_requests(state)

    def reservation_inbox(self, state: WorldState) -> dict[str, Any]:
        self._refresh_contingency_staleness(state)
        result = super().reservation_inbox(state)
        if self._v38_proactive_contingency():
            if self._v39_contingency_actor_clarity():
                updates = [
                    self._public_contingency_report(row, actor_clarity=True)
                    for row in state.reservation_contingency_reports
                    if row.response_day == state.day
                ]
                actionable = [
                    row for row in updates if row["status"] == "active_actionable"
                ]
                result["contingency_updates_today"] = updates
                result["actionable_contingency_reports"] = actionable
                result["actionable_contingency_report_count"] = len(actionable)
                result["reservation_actions_due_today"] = (
                    int(result.get("request_count", 0))
                    + int(result.get("plan_confirmation_count", 0))
                    + int(result.get("change_count", 0))
                    + int(result.get("risk_alert_count", 0))
                    + int(result.get("service_offer_count", 0))
                    + len(actionable)
                )
                return result
            reports = [
                self._public_contingency_report(row)
                for row in state.reservation_contingency_reports
                if row.status == "active" and row.response_day == state.day
            ]
            result["contingency_reports"] = reports
            result["contingency_report_count"] = len(reports)
            result["reservation_actions_due_today"] = (
                int(result.get("request_count", 0))
                + int(result.get("plan_confirmation_count", 0))
                + int(result.get("change_count", 0))
                + int(result.get("risk_alert_count", 0))
                + int(result.get("service_offer_count", 0))
                + len(reports)
            )
        return result

    def observation(self, state: WorldState, *, consume_inbox: bool):
        self._refresh_contingency_staleness(state)
        result = super().observation(state, consume_inbox=consume_inbox)
        if self._v38_proactive_contingency():
            count = sum(
                row.status
                == (
                    "active_actionable"
                    if self._v39_contingency_actor_clarity()
                    else "active"
                )
                and row.response_day == state.day
                for row in state.reservation_contingency_reports
            )

            def add(value: Any) -> None:
                if isinstance(value, dict):
                    if "reservation_actions_due_today" in value:
                        value[
                            (
                                "actionable_contingency_report_count"
                                if self._v39_contingency_actor_clarity()
                                else "reservation_contingency_report_count"
                            )
                        ] = count
                        value["reservation_actions_due_today"] += count
                    for child in tuple(value.values()):
                        add(child)
                elif isinstance(value, list):
                    for child in value:
                        add(child)

            add(result)
        return result

    @classmethod
    def _public_reservation_commitment(cls, commitment: Any) -> dict[str, Any]:
        result = super()._public_reservation_commitment(commitment)
        result.pop("version_recovery_offer_ids_by_line", None)
        return result

    @staticmethod
    def _contingency_eligibility_view(
        state: WorldState, commitment: Any | None
    ) -> dict[str, Any]:
        if commitment is None:
            return {
                "commitment_version": None,
                "contingency_inquiry_used": None,
                "contingency_submission_deadline": None,
                "contingency_eligible": False,
                "contingency_ineligible_reason": "no_commitment",
            }
        deadline = int(commitment.pickup_day) - MIN_NOTICE_DAYS
        reason = None
        if commitment.assignments or commitment.status in {
            "active",
            "completed",
            "returned",
        }:
            reason = "pickup_already_started"
        elif commitment.status != "confirmed":
            reason = "commitment_not_confirmed"
        elif commitment.contingency_inquiry_used:
            reason = "inquiry_already_used"
        elif state.day > deadline:
            reason = "submission_deadline_passed"
        return {
            "commitment_version": int(commitment.commitment_version),
            "contingency_inquiry_used": bool(
                commitment.contingency_inquiry_used
            ),
            "contingency_submission_deadline": deadline,
            "contingency_eligible": reason is None,
            "contingency_ineligible_reason": reason,
        }

    @staticmethod
    def _contingency_same_profile_offer_ids(
        request: Any, line: Any
    ) -> list[str]:
        profile_id = str(line.promise_profile_id or "")
        current_offer_id = str(line.selected_offer_id or line.offer_id)
        return sorted(
            offer_id
            for offer_id in request.inquiry_profile_member_ids_by_line.get(
                line.line_id, {}
            ).get(profile_id, [])
            if offer_id != current_offer_id
        )

    def _augment_v39_reservation_view(
        self, state: WorldState, result: dict[str, Any]
    ) -> None:
        requests = {row.request_id: row for row in state.reservation_requests}
        commitments = {
            row.commitment_id: row for row in state.reservation_commitments
        }
        for row in result.get("reservations", []):
            commitment = commitments.get(str(row.get("commitment_id") or ""))
            row.update(self._contingency_eligibility_view(state, commitment))
        for record in result.get("records", []):
            commitment = commitments.get(
                str(record.get("commitment_id") or "")
            )
            current_terms = record.get("current_terms")
            if not isinstance(current_terms, dict):
                continue
            current_terms.update(
                self._contingency_eligibility_view(state, commitment)
            )
            if commitment is None:
                continue
            request = requests[commitment.request_id]
            lines = {line.line_id: line for line in commitment.lines}
            for public_line in current_terms.get("locked_lines", []):
                line = lines[str(public_line["line_id"])]
                public_line.update(
                    current_offer_id=(line.selected_offer_id or line.offer_id),
                    promise_profile_id=line.promise_profile_id,
                    contingency_same_profile_offer_ids=(
                        self._contingency_same_profile_offer_ids(request, line)
                    ),
                )

    def view_reservation(self, state: WorldState, identifiers: list[str], **kwargs: Any):
        self._refresh_contingency_staleness(state)
        if not self._v38_proactive_contingency():
            return super().view_reservation(state, identifiers, **kwargs)
        wanted = set(map(str, identifiers))
        report_to_commitment = {
            row.report_id: row.commitment_id
            for row in state.reservation_contingency_reports
        }
        inquiry_to_commitment = {
            row.inquiry_id: row.commitment_id
            for row in state.reservation_contingency_inquiries
        }
        mapped_identifiers = []
        for identifier in map(str, identifiers):
            mapped = report_to_commitment.get(
                identifier, inquiry_to_commitment.get(identifier, identifier)
            )
            if mapped not in mapped_identifiers:
                mapped_identifiers.append(mapped)
        result = super().view_reservation(state, mapped_identifiers, **kwargs)
        if self._v39_contingency_actor_clarity():
            self._augment_v39_reservation_view(state, result)
        matching_commitments = {
            row.commitment_id
            for row in state.reservation_commitments
            if row.commitment_id in wanted
            or row.request_id in wanted
            or row.commitment_id in mapped_identifiers
        }
        history = [
            self._public_contingency_report(
                report, actor_clarity=self._v39_contingency_actor_clarity()
            )
            for report in state.reservation_contingency_reports
            if report.commitment_id in matching_commitments
            or report.report_id in wanted
            or report.inquiry_id in wanted
        ]
        if history:
            result["contingency_history"] = history
        return result

    def public_operating_rules(self) -> dict[str, Any]:
        result = super().public_operating_rules()
        if self._v38_proactive_contingency():
            result["proactive_reservation_contingency"] = {
                "eligibility": (
                    "One successful inquiry per confirmed pre-pickup commitment; "
                    "submit no later than pickup_day - 2."
                ),
                "proposal_scope": (
                    "One or two exact proposals, 2-14 days ahead, preserving duration. "
                    "Equipment may change only to an exact SKU inside the current Promise Profile."
                ),
                "financial_terms": "Locked rent, prepayment, and deposit do not change.",
                "customer_response": (
                    "The response arrives next morning and expires at that day's close. "
                    "Accepted proposals and a possible date-only counteroffer require explicit application."
                ),
                "feasibility": (
                    "Customer evaluation never checks store inventory, capacity, cash, "
                    "other commitments, or future realized weather."
                ),
                "cancellation": (
                    "A customer cancellation induced by contact uses the same refund and "
                    "compensation terms as manager cancellation and has no separate reputation effect."
                ),
                **(
                    {
                        "actor_interface": "updates_and_actionable_reports_v1",
                        "reservation_view": (
                            "Index and current_terms expose the current version, "
                            "eligibility, pickup_day - 2 deadline, and ineligibility "
                            "reason. Current line detail exposes legal same-Profile "
                            "exact-SKU proposal IDs without feasibility advice."
                        ),
                        "response_routing": (
                            "All reports appear in contingency_updates_today. Only "
                            "active_actionable reports also appear in "
                            "actionable_contingency_reports and expire if not applied."
                        ),
                        "public_observation_window": (
                            "weather and capacity tools support today plus fourteen "
                            "future days"
                        ),
                    }
                    if self._v39_contingency_actor_clarity()
                    else {}
                ),
            }
        return result
