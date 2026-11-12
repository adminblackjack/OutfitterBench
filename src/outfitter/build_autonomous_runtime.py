"""Build an immutable V47 candidate from V46 without touching legacy datasets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .channel_demand import ChannelDemandModel
from .return_dynamics import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .autonomous_rentals import ENVIRONMENT_PROTOCOL,TOOL_INTERFACE_VERSION,CORE_RELEASE_ID,POLICY_ID


def build_autonomous_runtime(source_runtime,output_dir):
    source,output=Path(source_runtime).resolve(),Path(output_dir).resolve()
    dataset=OutfitterDataset.load(source)
    if dataset.metadata.get("environment_protocol")!=SOURCE_PROTOCOL:
        raise ValueError("V47 requires a V46 source dataset")
    if output.exists() or output==source or source in output.parents or output in source.parents:
        raise ValueError("output must be a new independent directory")
    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in {"metadata.json","report.md","result.json","channel_demand_model.json"}:
            shutil.copy2(path,output/path.name)
    model=json.loads((source/"channel_demand_model.json").read_text())
    model.update(request_budget_bounds=[.6,1.6],rental_request_cash_salience=4.2,rental_budget_tail_coefficient=1.)
    ChannelDemandModel.from_dict(model)
    (output/"channel_demand_model.json").write_text(json.dumps(model,indent=2,sort_keys=True)+"\n")
    metadata=copy.deepcopy(dataset.metadata)
    metadata.pop("return_dynamics_policy",None); metadata.pop("return_delay_scale",None)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL,tool_interface_version=TOOL_INTERFACE_VERSION,
        core_release_id=CORE_RELEASE_ID,autonomous_rentals_policy=POLICY_ID,source_runtime=SOURCE_PROTOCOL,
        release_status="v47-autonomous-candidate-unvalidated", inherited_calibration_is_v47_evidence=False)
    result={"environment_protocol":ENVIRONMENT_PROTOCOL,"source_runtime":str(source),"validated_profitability":False}
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    metadata["content_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name!="metadata.json"}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    OutfitterDataset.load(output)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime",required=True); parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    print(json.dumps(build_autonomous_runtime(args.source_runtime,args.output_dir),indent=2))
