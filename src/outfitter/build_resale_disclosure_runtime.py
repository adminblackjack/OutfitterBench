"""Create independent V52 data; only version/disclosure metadata changes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .resale_disclosure import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, TOOL_INTERFACE_VERSION, POLICY_ID


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = OutfitterDataset.load(source)
    if (data.metadata.get('environment_protocol') != 'outfitter-seasonal-v51-reservation-overlap'
            or data.metadata.get('core_release_id') != 'outfitter-v51-reservation-overlap-core-r2'):
        raise ValueError('V52 requires a V51 core r2 source')
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('Output must be a new independent directory')
    shutil.copytree(source, output)
    metadata = copy.deepcopy(data.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, core_release_id=CORE_RELEASE_ID,
        tool_interface_version=TOOL_INTERFACE_VERSION, resale_formula_disclosure_policy=POLICY_ID,
        release_status='v52-public-resale-formula', source_runtime=data.metadata['environment_protocol'])
    metadata['content_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != 'metadata.json'}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    OutfitterDataset.load(output)
    return {'runtime':str(output),'environment_protocol':ENVIRONMENT_PROTOCOL,
            'economic_payloads_changed':False,'new_product_quotes':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(build(a.source,a.output),ensure_ascii=False))
