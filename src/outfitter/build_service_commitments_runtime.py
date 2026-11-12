"""Independent V32 projection; frozen V31 catalog/economics are copied unchanged."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from .data import OutfitterDataset
from .build_warehouse_capacity_runtime import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .service_commitments import ENVIRONMENT_PROTOCOL, TOOL_INTERFACE_VERSION, POLICY


def build_service_commitments_runtime(source_runtime, output_dir):
    source, output = Path(source_runtime).resolve(), Path(output_dir).resolve()
    data = OutfitterDataset.load(source)
    if (data.metadata.get("environment_protocol"), data.metadata.get("tool_interface_version")) != (SOURCE_PROTOCOL, 27):
        raise ValueError("V32 requires the frozen V31 runtime and interface 27")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("source and output must be independent directories")
    if output.exists():
        raise ValueError("output must not already exist; never overwrite a runtime")
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name not in {"metadata.json", "result.json", "report.md"}:
            shutil.copy2(path, output / path.name)
    metadata = copy.deepcopy(data.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, tool_interface_version=TOOL_INTERFACE_VERSION,
        release_status="v32-candidate-awaiting-deterministic-and-static-checks", source_runtime=str(source),
        service_commitment_policy=copy.deepcopy(POLICY))
    result = {"format":"outfitter-service-commitments-runtime-builder-v1", "environment_protocol":ENVIRONMENT_PROTOCOL,
        "source_runtime":str(source), "output_runtime":str(output), "warehouse_capacity_limit":200,
        "underlying_customer_parameter_files_changed":False,
        "customer_contract_choice_semantics_changed":True, "market_reference_is_empirical":False,
        "three_mechanisms_are_independently_configurable":True}
    (output / "result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (output / "report.md").write_text(
        "# V32 service commitment candidate\n\n"
        "Independent opt-in date flexibility, reschedule decline outcomes, and fixed-reward service offers. "
        "V31 warehouse N=200 and all underlying catalog, preference/WTP, feedback, wear, fees and valuation data remain unchanged. "
        "New market quote formula is a synthetic design, not observed market prices. "
        "Fixed offers are filtered against the unchanged customer kernel at creation, never at manager acceptance. "
        "This runtime is a candidate, not evidence of LLM economic separation.\n")
    metadata["content_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != "metadata.json"}
    (output / "metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    OutfitterDataset.load(output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    print(json.dumps(build_service_commitments_runtime(args.source_runtime,args.output_dir),indent=2))


if __name__ == "__main__":
    main()
