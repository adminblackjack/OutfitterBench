"""Build V50 from a frozen V49 r2 runtime, without modifying the input."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .school_holiday import (CORE_RELEASE_ID as SOURCE_CORE,
                             ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL,
                             SEASONAL_ANCHORS, policy_document as school_policy)
from .summer_pressure import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, TOOL_INTERFACE_VERSION, policy_document


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = OutfitterDataset.load(source)
    if (data.metadata.get('environment_protocol') != SOURCE_PROTOCOL
            or data.metadata.get('core_release_id') != SOURCE_CORE
            or data.metadata.get('school_holiday_policy') != school_policy()
            or data.market_rules.seasonal_anchors != SEASONAL_ANCHORS):
        raise ValueError('V50 requires the frozen V49 core-r2 source')
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('Output must be a new independent directory')
    shutil.copytree(source, output)
    metadata = copy.deepcopy(data.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, core_release_id=CORE_RELEASE_ID,
        tool_interface_version=TOOL_INTERFACE_VERSION, summer_pressure_policy=policy_document(),
        source_runtime=SOURCE_PROTOCOL, release_status='v50-d2-scheduling-pressure-candidate')
    # Preserve payloads byte-for-byte. result.json exists only in research
    # runtimes; portable releases do not need it.
    if (output/'result.json').exists():
        (output/'result.json').write_text(json.dumps({'environment_protocol': ENVIRONMENT_PROTOCOL,
            'source_runtime': SOURCE_PROTOCOL, 'profile': 'D2', 'validated_model_discrimination': False},indent=2)+'\n')
    metadata['content_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != 'metadata.json'}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    OutfitterDataset.load(output)
    return {'runtime': str(output), 'protocol': ENVIRONMENT_PROTOCOL, 'profile': 'D2'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(build(args.source,args.output)))
