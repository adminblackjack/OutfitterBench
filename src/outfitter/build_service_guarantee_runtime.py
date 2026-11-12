"""Create V48 from an immutable V47 runtime; never rewrite old data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .data import OutfitterDataset
from .service_guarantee import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, TOOL_INTERFACE_VERSION, POLICY_ID


def build(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    dataset=OutfitterDataset.load(source)
    if dataset.metadata.get('environment_protocol')!='outfitter-seasonal-v47-autonomous-rentals':
        raise ValueError('V48 requires a V47 source runtime')
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('output must be a new independent runtime directory')
    shutil.copytree(source,output)
    metadata=copy.deepcopy(dataset.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL,core_release_id=CORE_RELEASE_ID,
        tool_interface_version=TOOL_INTERFACE_VERSION,service_guarantee_policy=POLICY_ID,
        release_status='v48-service-guarantee-candidate',source_runtime='outfitter-seasonal-v47-autonomous-rentals')
    metadata['reservation_failure_settlement']={'format':POLICY_ID,'first_day_multiplier':1,
        'notice_decay_days':7,'notice_floor':.25,'basis':'original_confirmation_whole_line_first_day'}
    metadata.setdefault('reservation_mechanism',{})['failure_settlement_policy']=POLICY_ID
    (output/'result.json').write_text(json.dumps({'environment_protocol':ENVIRONMENT_PROTOCOL,
        'source_runtime':'outfitter-seasonal-v47-autonomous-rentals','validated_profitability':False},indent=2)+'\n')
    metadata['content_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name!='metadata.json'}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    OutfitterDataset.load(output)
    return {'runtime':str(output),'protocol':ENVIRONMENT_PROTOCOL}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.source,a.output)))
