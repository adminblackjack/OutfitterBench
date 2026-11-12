"""V47 fixed policies. Operational facts enter only through public RPC."""
from __future__ import annotations

import copy

from .scripted_merchant import money
from .simple_policy_stress import PRICES,project,direct_margin
from .v45_policy_experiment import V45Merchant

FIXED_POLICIES=("C","U2","U4","G","HP-1.5","HP-2","HP-3","HP-mixed")


class AutonomousControl(V45Merchant):
    def __init__(self,mode,rpc):
        if mode not in FIXED_POLICIES:
            raise ValueError(mode)
        scale={"HP-1.5":1.5,"HP-2":2.,"HP-3":3.}.get(mode,1.25)
        super().__init__("G" if mode=="G" else "C",rpc,scale)
        self.mode=mode
        self.buffer_days={"U2":2,"U4":4}.get(mode,0)
        self.quote_prices={oid:money(price*2.) for oid,price in PRICES.items()} if mode=="HP-mixed" else dict(self.prices)
        self.episodes=[]

    def physical(self):
        self.status,rented,self.inbox=self.many([
            ("business_status",{}),("view_active_rentals",{}),("reservation_inbox",{})])
        self.rentals=rented["rentals"]
        self.episodes=rented.get("rental_episodes",[])
        self.assets=self.pages("view_rental_assets",{"view":"asset","limit":50},"assets")
        self.jobs=self.pages("view_rental_capacity",{"view":"service_jobs","limit":10},"jobs")

    def load(self):
        super().load()
        missing=sorted({ep["commitment_id"] for ep in self.episodes
                        if ep["status"]=="active" and ep.get("commitment_id") and ep["commitment_id"] not in self.records})
        self.load_record(missing)

    def snapshot(self):
        value=super().snapshot()
        value["rentals"]=copy.deepcopy(self.rentals)
        for row in value["rentals"]:
            # This is this policy's forecast, not an environment guarantee.
            row["estimated_return_day"]=max(self.status["day"]+1,
                row["return_day"]+self.buffer_days,row.get("estimated_return_day") or row["return_day"])
        return value

    def assess(self,candidate,exclude=None):
        future=self.future(exclude)
        padded=copy.deepcopy(future+[candidate])
        for row in padded:
            row["return"]+=self.buffer_days
        snapshot=self.snapshot()
        before=project(snapshot,padded[:-1]); after=project(snapshot,padded)
        okay=candidate["id"] not in after["failures"] and not(set(after["failures"])-set(before["failures"]))
        margin=direct_margin(snapshot,candidate,after)
        return {"accept":okay and margin>0,"margin":margin,"missing_optional":[],"failures":after["failures"]}

    def quote_options(self,request):
        posted=self.prices
        self.prices=self.quote_prices
        try:
            return super().quote_options(request)
        finally:
            self.prices=posted

    def change(self,change):
        if change["change_type"]!="reschedule":
            raise AssertionError("V47 must not expose extension approvals")
        return super().change(change)

    def protect(self):
        original=self.records
        self.records=copy.deepcopy(original)
        try:
            for record in self.records.values():
                terms=record["current_terms"]
                if terms["status"] in {"confirmed","at_risk"}:
                    terms["return_day"]+=self.buffer_days
            return super().protect()
        finally:
            self.records=original
