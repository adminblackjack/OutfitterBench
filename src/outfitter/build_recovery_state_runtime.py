"""Create a V54 snapshot without changing V53 economic payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .operating_history import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL
from .recovery_state import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, POLICY_ID, TOOL_INTERFACE_VERSION


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = OutfitterDataset.load(source)
    if data.metadata.get('environment_protocol') != SOURCE_PROTOCOL:
        raise ValueError('V54 requires a V53 source snapshot')
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('output must be a new independent directory')
    shutil.copytree(source, output)
    metadata = dict(data.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, core_release_id=CORE_RELEASE_ID,
                    tool_interface_version=TOOL_INTERFACE_VERSION, recovery_state_policy=POLICY_ID,
                    package_version='0.54.0', source_runtime=SOURCE_PROTOCOL,
                    release_status='v54-recovery-state')
    # Preserve economic payloads verbatim, including the prior build provenance.
    metadata['content_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != 'metadata.json'}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    OutfitterDataset.load(output)
    return metadata


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(build(args.source,args.output)['environment_protocol'])
