"""Autonomous physical returns, optional customer reports, and ordinary rent."""
from __future__ import annotations

import copy
from dataclasses import replace

from ..autonomous_rentals import (POLICY_ID,STREAM,MAX_EXTRA_DAYS,NOTICE_MODES,NOTICE_WEIGHTS,
    PUBLIC_RULES,continuation_probability)
from ..channel_demand import effective_rental_cash_segment
from ..scripted_merchant import money
from ..autonomous_rentals import LEAD_BANDS,LEAD_WEIGHTS
from .common import _ARCHETYPE_INDEX,_RNG_ARRIVALS,_RNG_PARTY_COMPOSITION,_RNG_CUSTOMER_BASKET,_RNG_RESERVATION_ROUTE
from ..v5 import seasonal_index
from ..weather_information import POTENTIAL_POOL_MULTIPLIERS
from ..reservation_difficulty import TARIFF_RULE


class AutonomousRentalsMixin:
    def reset(self):
        state=super().reset()
        if self._v47_autonomous_rentals():
            state.rental_episode_protocol=POLICY_ID
        return state

    def _v47_assert_state_version(self,state):
        if self._v47_autonomous_rentals() and state.rental_episode_protocol!=POLICY_ID:
            raise ValueError("V47 requires a fresh V47 state; older rental states cannot be resumed")

    def _v47_autonomous_rentals(self):
        return (self.dataset.metadata.get("autonomous_rentals_policy")==POLICY_ID
                and int(self.dataset.metadata.get("tool_interface_version",0))>=42)

    def _v47_rng(self, address, identity):
        return self._stable_event_rng(STREAM,*address,identity=identity)

    def respond_reservation_changes(self,state,decisions):
        if self._v47_autonomous_rentals():
            changes={row.change_id:row for row in state.reservation_change_requests}
            if any(row.get("change_id") in changes and changes[row["change_id"]].change_type!="reschedule"
                   for row in decisions):
                raise ValueError("V47 has no active-trip extension approval or refusal")
        return super().respond_reservation_changes(state,decisions)

    def _v47_customer_shape(self,pickup_day,archetype,index):
        address=(pickup_day,_ARCHETYPE_INDEX[archetype],index)
        party=self._event_rng(_RNG_PARTY_COMPOSITION,*address)
        adults=int(party.integers(*{"overnight":(1,4),"picnic":(2,5),"lightweight":(1,2)}[archetype]))
        day_type=self._calendar(pickup_day)["day_type"]
        chance={"weekday":.08,"weekend":.15,"public_holiday":.35,"school_holiday":.45}[day_type]
        children=int(party.integers(1,3)) if archetype!="lightweight" and party.random()<chance else 0
        basket=self._event_rng(_RNG_CUSTOMER_BASKET,*address)
        if archetype=="overnight":
            duration=int(basket.choice([2,3,4,5],p=[.12,.32,.36,.20])) if day_type in {"public_holiday","school_holiday"} else int(basket.choice([1,2,3,4],p=[.22,.40,.28,.10]))
        elif archetype=="picnic":
            duration=int(basket.choice([1,2],p=[.82,.18])) if day_type in {"weekend","public_holiday","school_holiday"} else 1
        else:
            duration=int(basket.choice([1,2,3],p=[.58,.34,.08]))
        return {"adults":adults,"children":children,"duration":duration,
                "basecamp":archetype=="overnight" and self._v47_rng(address,"basecamp").random()<.30}

    def _reservation_route(self,pickup_day,archetype_index,customer_index):
        if not self._v47_autonomous_rentals():
            return super()._reservation_route(pickup_day,archetype_index,customer_index)
        if not self.config.reservations_enabled:
            return False,0
        archetype=next(key for key,value in _ARCHETYPE_INDEX.items() if value==archetype_index)
        if archetype=="supply":
            return False,0
        shape=self._v47_customer_shape(pickup_day,archetype,customer_index)
        kind="picnic_group" if archetype=="picnic" and (shape["children"] or shape["adults"]>=4) else archetype
        weights=list(LEAD_WEIGHTS[kind])
        if self._calendar(pickup_day)["day_type"]=="public_holiday":
            shift=min(.10,weights[0]-.05); weights[0]-=shift; weights[2]+=shift
        rng=self._v47_rng((pickup_day,archetype_index,customer_index),"booking-lead")
        allowed=[day for lo,hi in LEAD_BANDS for day in range(lo,hi+1)
                 if self.config.reservation_booking_lead_min_days<=day<=self.config.reservation_booking_lead_max_days]
        probabilities=[weights[next(i for i,(lo,hi) in enumerate(LEAD_BANDS) if lo<=day<=hi)] /
                       (next(hi-lo+1 for lo,hi in LEAD_BANDS if lo<=day<=hi)) for day in allowed]
        total=sum(probabilities)
        lead=int(rng.choice(allowed,p=[p/total for p in probabilities]))
        selected=self._event_rng(_RNG_RESERVATION_ROUTE,pickup_day,archetype_index,customer_index).random()<self.config.reservation_share
        return bool(selected and pickup_day-lead>=1),lead

    def _v47_base_arrival_rate(self,day,kind):
        calendar=self._calendar(day)
        rate=self._calendar_arrival_lambda(kind,calendar)*self.rules.arrival_scale*self.config.arrival_scale_multiplier
        rate*=seasonal_index(self._source_weather_day(day),self.rules.seasonal_anchors)*POTENTIAL_POOL_MULTIPLIERS[kind]
        peak=float(self.dataset.metadata.get("reservation_calendar_peak_multiplier",1.))
        rate*=peak if calendar["day_type"]=="public_holiday" or calendar["is_weekend"] else max(.35,(7-2*peak)/5)
        return rate

    def _v47_calendar_weight(self,day):
        calendar=self._calendar(day)
        if calendar["is_weekend"] or calendar["day_type"]=="public_holiday":
            return 1.6
        return 1.2 if self._calendar(day+1)["is_weekend"] else .8

    def _v47_arrival_rate(self,day,kind):
        if not hasattr(self,"_v47_rate_normalizers"):
            self._v47_rate_normalizers={}
        if kind not in self._v47_rate_normalizers:
            rates=[self._v47_base_arrival_rate(d,kind) for d in range(1,self.config.max_days+1)]
            weighted=sum(r*self._v47_calendar_weight(d) for d,r in enumerate(rates,1))
            self._v47_rate_normalizers[kind]=sum(rates)/weighted
        return self._v47_base_arrival_rate(day,kind)*self._v47_calendar_weight(day)*self._v47_rate_normalizers[kind]

    def _potential_arrival_count(self,pickup_day,archetype):
        if not self._v47_autonomous_rentals() or archetype=="supply":
            return super()._potential_arrival_count(pickup_day,archetype)
        return int(self._event_rng(_RNG_ARRIVALS,pickup_day,_ARCHETYPE_INDEX[archetype]).poisson(
            self._v47_arrival_rate(pickup_day,archetype)))

    def _reservation_bundle_sessions(self,state,primary,*,pickup_day,booking_lead_days):
        sessions=super()._reservation_bundle_sessions(state,primary,pickup_day=pickup_day,booking_lead_days=booking_lead_days)
        if (not self._v47_autonomous_rentals() or not self.config.reservation_bundles_enabled
                or not self._v47_customer_shape(pickup_day,primary.archetype,int(primary.event_address[2]))["basecamp"]):
            return sessions
        sessions=[(session,required) for session,required in sessions if required]
        future=copy.copy(state); future.day=pickup_day
        forecast=next(row for row in self.weather_forecast(state,pickup_day-state.day+1) if row["day"]==pickup_day)
        chair=self._make_choice_session(future,service_key="rental:camp_chair",units_requested=primary.adults+primary.children,
            duration_days=primary.duration_days,adults=primary.adults,children=primary.children,size_class="n/a",
            archetype=primary.archetype,weather=forecast["condition"],request_index=3,
            event_address=primary.event_address[:3],intent_role="required",trip_weather_context=primary.trip_weather_context)
        chair=replace(chair,pickup_day=pickup_day,booking_lead_days=booking_lead_days,
            forecast_adversity=primary.forecast_adversity,forecast_uncertainty=primary.forecast_uncertainty)
        return sessions+[(self._reservation_customer_coherent_session(primary,chair),True)]

    def _v47_register(self,state,contracts,session,*,commitment=None):
        if not contracts:
            return
        address=tuple(session.event_address[:3])
        episode_id=(f"trip-{commitment.commitment_id}" if commitment else
                    f"trip-walkin-{address[0]:03d}-{address[1]:02d}-{address[2]:04d}")
        if episode_id not in state.rental_episodes:
            model=self.channel_demand_model
            budget=effective_rental_cash_segment(model,model.segments[session.customer_segment],
                duration_days=session.duration_days,party_equivalent=max(1.,session.adults+.6*session.children),
                request_budget_multiplier=session.request_budget_multiplier).cash_scale_cny
            state.rental_episodes[episode_id]={
                "episode_id":episode_id,"commitment_id":commitment.commitment_id if commitment else None,
                "source":"reservation" if commitment else "walk_in","start_day":state.day,
                "contractual_return_day":contracts[0].approved_return_day or contracts[0].return_day,
                "contract_ids":[],"daily_rates":{},"extra_days":0,"accrued_rent":0.,"collected_rent":0.,
                "status":"active","reported_return_day":None,"actual_return_day":None,"notices":[],
                "last_processed_day":None,"return_today":None,
                "private":{"address":list(address),"archetype":session.archetype,"cash_scale":budget,
                    "flexibility":(next(r.schedule_flexibility for r in state.reservation_requests
                        if r.request_id==commitment.request_id) if commitment else
                        float(self._v47_rng(address,"walkin-flexibility").uniform(.15,.95))),
                    "notice_mode":str(self._v47_rng(address,"notice-mode").choice(NOTICE_MODES,p=NOTICE_WEIGHTS)),
                    "initial_notice_sent":False}}
        ep=state.rental_episodes[episode_id]
        allocations={}
        if commitment:
            _,deltas,_=self._extension_rent_delta(commitment,
                proposed_return_day=commitment.return_day+1,bill_fulfilled_only=True)
            allocations=self._v32_extension_allocations(state,commitment,deltas)
        for contract in contracts:
            if contract.contract_id in ep["contract_ids"]:
                continue
            if contract.start_day!=ep["start_day"] or contract.return_day!=ep["contractual_return_day"]:
                raise AssertionError("one trip must have one initial date interval")
            contract.rental_episode_id=episode_id
            contract.estimated_return_day=None
            ep["contract_ids"].append(contract.contract_id)
            ep["daily_rates"][contract.contract_id]=(allocations[contract.contract_id] if commitment
                                                       else money(state.prices[contract.offer_id]*.5))
        if commitment:
            commitment.estimated_return_day=None

    def _rent_from_offer(self,state,session,offer_id,remaining_units,outcomes):
        before=len(state.contracts)
        result=super()._rent_from_offer(state,session,offer_id,remaining_units,outcomes)
        if self._v47_autonomous_rentals():
            self._v47_register(state,state.contracts[before:],session)
        return result

    def _process_reservation_pickups(self,state,**kwargs):
        super()._process_reservation_pickups(state,**kwargs)
        if not self._v47_autonomous_rentals() or state.done:
            return
        requests={r.request_id:r for r in state.reservation_requests}
        for c in state.reservation_commitments:
            if c.status!="active" or f"trip-{c.commitment_id}" in state.rental_episodes:
                continue
            request=requests[c.request_id]
            primary=next(line for line in request.lines if line.required)
            session=self._restore_quote_context(request.quote_context_by_line[primary.line_id])
            contracts=[r for r in state.contracts if r.reservation_id==c.commitment_id and r.status=="active"]
            self._v47_register(state,contracts,session,commitment=c)
        self._v47_publish_notices(state)

    def _v47_stays(self,ep,day,condition):
        due=ep["contractual_return_day"]
        p=ep["private"]
        probability=continuation_probability(archetype=p["archetype"],flexibility=p["flexibility"],
            condition=condition,calendar=self._calendar(day),next_calendar=self._calendar(day+1),
            daily_rent=sum(ep["daily_rates"].values()),cash_scale=p["cash_scale"],extra_days=day-due)
        draw=self._v47_rng(p["address"],f"continue:{day}").random()
        return bool(draw<probability)

    def _v47_report(self,state,ep):
        due=ep["contractual_return_day"]
        last=min(state.max_days,due+MAX_EXTRA_DAYS)
        forecasts={r["day"]:r for r in self.weather_forecast(state,last-state.day+1)}
        first=max(due,state.day+(1 if ep["last_processed_day"]==state.day else 0))
        estimate=last
        for day in range(first,last+1):
            from ..weather_information import forecast_regime
            if not self._v47_stays(ep,day,forecast_regime(forecasts[day])):
                estimate=day
                break
        ep["reported_return_day"]=estimate
        ep["notices"].append({"day":state.day,"type":"customer_return_report", "estimated_return_day":estimate,
                               "forecast_issued_day":state.day,"is_guarantee":False})
        for contract in state.contracts:
            if contract.rental_episode_id==ep["episode_id"]:
                contract.estimated_return_day=estimate
        if ep["commitment_id"]:
            c=next(c for c in state.reservation_commitments if c.commitment_id==ep["commitment_id"])
            c.estimated_return_day=estimate
        state.inbox.append(f"租赁行程 {ep['episode_id']} 报告预计第{estimate}日归还；这不是归还保证。")

    def _v47_publish_notices(self,state):
        for ep in state.rental_episodes.values():
            if ep["status"]!="active" or ep["return_today"]==state.day:
                continue
            private=ep["private"]
            mode=private["notice_mode"]
            if mode=="silent" or any(n["day"]==state.day and n["type"]=="customer_return_report" for n in ep["notices"]):
                continue
            due=ep["contractual_return_day"]
            at={"early":max(ep["start_day"],due-2),"due":due,"overdue":due+1}[mode]
            if not private["initial_notice_sent"] and state.day>=at:
                private["initial_notice_sent"]=True
                self._v47_report(state,ep)
            elif (ep["reported_return_day"] is not None and state.day>=ep["reported_return_day"]
                    and self._v47_rng(private["address"],f"revise-notice:{state.day}").random()<.35):
                self._v47_report(state,ep)

    def _v47_prepare_returns(self,state):
        contracts={r.contract_id:r for r in state.contracts}
        assets={a.asset_id:a for a in state.assets}
        for ep in state.rental_episodes.values():
            due=ep["contractual_return_day"]
            if ep["status"]!="active" or state.day<due or ep["last_processed_day"]==state.day:
                continue
            ep["last_processed_day"]=state.day
            if state.day<state.max_days and self._v47_stays(ep,state.day,self._demand_weather_regime(state.day)):
                ep["extra_days"]+=1
                daily=money(sum(ep["daily_rates"].values()))
                ep["accrued_rent"]=money(ep["accrued_rent"]+daily)
                for cid in ep["contract_ids"]:
                    r=contracts[cid]; asset=assets[r.asset_id]
                    r.duration_days+=1
                    # Internal elapsed-use endpoint, never an actor ETA.
                    r.return_day=state.day+1
                    asset.used_life_days+=1
                    self._refresh_contract_lifecycle_damage(asset,r)
                    self._charge_supplier_trial_contract_extension(state,asset,r,duration_days=1,
                        reason="供应商试用资产实际继续占用使用费")
                    self._v47_record_continuation_service(state, r, rental_days=1)
                ep["notices"].append({"day":state.day,"type":"not_returned","overdue_days":state.day-due+1})
            else:
                ep["return_today"]=state.day
                extra=ep["accrued_rent"]
                state.cash=money(state.cash+extra)
                state.cumulative_rental_revenue=money(state.cumulative_rental_revenue+extra)
                self._add(state.current_day_ledger,"rental_revenue",extra)
                ep["collected_rent"]=extra
                for cid in ep["contract_ids"]:
                    r=contracts[cid]
                    r.return_day=state.day
                    additional_rent=money(ep["daily_rates"][cid]*ep["extra_days"])
                    r.rent_paid=money(r.rent_paid+additional_rent)
                    if additional_rent:
                        self._v47_record_continuation_service(state, r, revenue=additional_rent)
                if ep["commitment_id"]:
                    c=next(c for c in state.reservation_commitments if c.commitment_id==ep["commitment_id"])
                    c.post_pickup_rent_collected=money(c.post_pickup_rent_collected+extra)
                    self._reservation_event(state,"continuation_rent_collected",request_id=c.request_id,
                        commitment_id=c.commitment_id,details={"additional_rent_collected":extra,
                            "continuation_rent_accrued":ep["accrued_rent"],"collection_timing":"physical_return",
                            "reason":"automatic_ordinary_continuation_settlement"})
                if ep["extra_days"]:
                    state.inbox.append(f"行程 {ep['episode_id']} 实际归还，正常续日租金 {extra:.2f}。")
        self._v47_publish_notices(state)

    def _v47_record_continuation_service(self, state, contract, *, revenue=0.0, rental_days=0):
        """Post actual extra asset-days and collected cash, never another sale."""
        from .. import operating_history
        if operating_history.enabled(self.dataset.metadata):
            operating_history.record_delivery(state, contract.offer_id,
                "reservation" if contract.reservation_id else "walk_in",
                revenue=revenue, rental_days=rental_days, continuation=True)
        product_type = contract.service_line
        metric = self._service_metric(state, self.service_key_for_product_type(product_type))
        empty = {"fulfilled_units": 0, "revenue": 0.0, "rental_days": 0,
                 "deposits_collected": 0.0, "retail_cogs": 0.0}
        product = metric["product_fulfillment_breakdown"].setdefault(product_type, dict(empty))
        offer = metric["offer_fulfillment_breakdown"].setdefault(
            contract.offer_id, {**empty, "product_type": product_type})
        for row in (metric, product, offer):
            row["revenue"] = money(row["revenue"] + revenue)
            row["continuation_rental_revenue"] = money(row.get("continuation_rental_revenue", 0) + revenue)
            row["continuation_rental_days"] = row.get("continuation_rental_days", 0) + rental_days
        for row in (product, offer):
            row["rental_days"] += rental_days

    def _v47_continuation_terms(self, state, commitment_id):
        ep = state.rental_episodes.get(f"trip-{commitment_id}", {})
        return {"continuation_rent_accrued": ep.get("accrued_rent", 0.0),
                "continuation_rent_collected": ep.get("collected_rent", 0.0),
                "extra_occupied_days": ep.get("extra_days", 0)}

    def view_reservation(self, state, identifiers, **kwargs):
        result = super().view_reservation(state, identifiers, **kwargs)
        if self._v47_autonomous_rentals():
            for record in result.get("records", []):
                if isinstance(record.get("current_terms"), dict):
                    record["current_terms"].update(
                        self._v47_continuation_terms(state, record.get("commitment_id")))
        return result

    def _process_morning(self,state):
        if self._v47_autonomous_rentals() and not state.done:
            self._v47_prepare_returns(state)
        result=super()._process_morning(state)
        if self._v47_autonomous_rentals():
            by_id={r.contract_id:r for r in state.contracts}
            for ep in state.rental_episodes.values():
                if ep["status"]=="active" and all(by_id[cid].status=="returned" for cid in ep["contract_ids"]):
                    ep["status"]="returned"; ep["actual_return_day"]=state.day
                    ep["notices"].append({"day":state.day,"type":"returned","actual_return_day":state.day})
                    if ep["commitment_id"] and ep["extra_days"]:
                        state.cumulative_reservation_late_returns+=1
        return result

    def _v47_return_due_today(self,state,contract):
        return state.rental_episodes[contract.rental_episode_id]["return_today"]==state.day

    @staticmethod
    def _v47_public_episode(ep,day):
        return {key:copy.deepcopy(ep[key]) for key in ("episode_id","source","commitment_id","contract_ids",
            "start_day","contractual_return_day","status","reported_return_day","actual_return_day",
            "extra_days","accrued_rent","collected_rent")} | {
                "overdue":ep["status"]=="active" and day>=ep["contractual_return_day"],
                "notices_today":[copy.deepcopy(n) for n in ep["notices"] if n["day"]==day],"return_date_is_guaranteed":False}

    def _v47_active_payload(self,state,rows):
        for row in rows:
            contract=next(r for r in state.contracts if r.contract_id==row["contract_id"])
            ep=state.rental_episodes[contract.rental_episode_id]
            row.update(rental_episode_id=ep["episode_id"],return_day=ep["contractual_return_day"],
                contractual_return_day=ep["contractual_return_day"],estimated_return_day=ep["reported_return_day"],
                return_date_is_guaranteed=False,overdue=state.day>=ep["contractual_return_day"])
        episodes=[self._v47_public_episode(ep,state.day) for ep in state.rental_episodes.values()
                  if ep["status"]=="active" or any(n["day"]==state.day for n in ep["notices"])]
        return {"rentals":rows,"rental_episodes":episodes}

    def business_manual(self):
        result=super().business_manual()
        if self._v47_autonomous_rentals():
            rules=[]
            for rule in result['reservations']['rules']:
                if rule==TARIFF_RULE:
                    rules.append(rule.replace('Extensions disclose the additional amount due under the locked terms.',
                        'Continued possession accrues ordinary additional-day rent under the locked terms, settled on physical return.'))
                elif rule.startswith('A customer may request a reschedule or extension, or cancel a trip.'):
                    rules.append(rule.split(' Accepting an extension')[0].replace('a reschedule or extension','a pre-pickup reschedule'))
                elif 'extension' not in rule.lower():
                    rules.append(rule)
            result['reservations']['rules']=rules
            result["reservations"]["rules"].append(PUBLIC_RULES)
            result["reservations"]["rules"].append(
                "total_locked_rent is the agreed contract amount and equals the sum of locked line rents. "
                "Autonomous continued use does not amend that amount or the contractual dates/version. "
                "Reservation current_terms separately reports continuation_rent_accrued, "
                "continuation_rent_collected and extra_occupied_days. Accrued rent is not cash until return; "
                "post_pickup_rent_collected includes the collected continuation rent.")
            result["finance"]["rules"].append(
                "service_history revenue includes continuation rent on its collection day. "
                "continuation_rental_revenue isolates that collected amount; continuation_rental_days "
                "counts extra asset-days on the day each occurs. Offer rental_days includes contracted "
                "asset-days booked at pickup plus those extra asset-days, not elapsed use alone. "
                "Continuation creates no new customer, transaction or fulfilled unit. History covers "
                "completed days only; a period's extra-day counts and receipts may occur on different dates.")
            if int(self.dataset.metadata.get("tool_interface_version", 0)) >= 44:
                result["reservations"]["rules"].append(
                    "resolve_reservation_risks validates actions as one atomic batch. An invalid action "
                    "undoes all business changes in that batch, including refunds, inquiry rights, "
                    "recovery selection and contract versions. Complete-plan and line-fallback paths "
                    "cannot be mixed. Real payment failure or bankruptcy is not a validation error: "
                    "its consequences remain and subsequent actions are skipped. Each materially "
                    "changed contract receives one new version per successful batch.")
                result["finance"]["rules"].append(
                    "Partial-pickup compensation is separate from missing-line rent credit. Closing "
                    "receipts report compensation_due, compensation_paid (paid in full), and "
                    "compensation_amount_paid from settlement; credit is not a second cash refund.")
                result["assets"]["rules"].append(
                    "view_rental_assets summary status filters select which SKUs appear, not the "
                    "denominator of their counts. Every returned SKU shows its full state distribution. "
                    "matched_assets counts filter matches; total_assets includes all historical asset "
                    "records. held_assets excludes sold and physically disposed assets; owned_assets "
                    "and supplier_trial_assets split that held total by ownership. inspection_due_assets "
                    "counts held equipment only. Asset detail view still filters individual assets.")
        return result

    def view_rental_capacity(self,state,**kwargs):
        result=super().view_rental_capacity(state,**kwargs)
        if not self._v47_autonomous_rentals() or result.get("view")!="timeline":
            return result
        selected=set(result["offers"])
        by_id={r.contract_id:r for r in state.contracts}
        rows=[]
        for ep in state.rental_episodes.values():
            if ep["status"]!="active":
                continue
            members=[by_id[cid] for cid in ep["contract_ids"] if by_id[cid].offer_id in selected]
            if members:
                rows.append({**self._v47_public_episode(ep,state.day),
                    "assets":[{"asset_id":r.asset_id,"offer_id":r.offer_id} for r in members]})
        result["unreturned_rental_episodes"]=rows
        # Legacy intervals end at an internal scheduler tick, not a promised ETA.
        # Replace them with observed occupancy, and leave the future unknown.
        result["walk_in_occupancy_intervals"]=[]
        result["interpretation"]+=" Unreturned episodes remain physically out. Future return dates are not guaranteed; contractual intervals do not establish future availability."
        return result
