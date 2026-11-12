"""Build an independent V49 runtime/config from V48; never mutate its inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .config import OutfitterConfig
from .data import OutfitterDataset
from .school_holiday import ENVIRONMENT_PROTOCOL, CORE_RELEASE_ID, TOOL_INTERFACE_VERSION, policy_document, SEASONAL_ANCHORS
from .service_guarantee import ENVIRONMENT_PROTOCOL as SOURCE_PROTOCOL


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    dataset = OutfitterDataset.load(source)
    if dataset.metadata.get('environment_protocol') != SOURCE_PROTOCOL:
        raise ValueError('V49 requires a V48 source runtime')
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError('Output must be a new independent directory')
    shutil.copytree(source, output)
    rules = dataset.market_rules.to_dict()
    rules['seasonal_anchors'] = [list(row) for row in SEASONAL_ANCHORS]
    (output / 'market_rules.json').write_text(json.dumps(rules, indent=2, sort_keys=True) + '\n')
    metadata = copy.deepcopy(dataset.metadata)
    metadata.update(environment_protocol=ENVIRONMENT_PROTOCOL, core_release_id=CORE_RELEASE_ID,
        tool_interface_version=TOOL_INTERFACE_VERSION, school_holiday_policy=policy_document(),
        release_status='v49-school-holiday-candidate', source_runtime=SOURCE_PROTOCOL)
    (output / 'result.json').write_text(json.dumps({'environment_protocol': ENVIRONMENT_PROTOCOL,
        'source_runtime': SOURCE_PROTOCOL, 'validated_profitability': False}, indent=2) + '\n')
    metadata['content_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != 'metadata.json'}
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    OutfitterDataset.load(output)
    return {'runtime': str(output), 'protocol': ENVIRONMENT_PROTOCOL}


def write_configs(config_dir, runtime):
    config_dir = Path(config_dir).resolve()
    for track in ('main', 'secondary'):
        target = config_dir / f'outfitter_v49_school_holiday_r2_{track}.yaml'
        if target.exists():
            raise ValueError(f'Will not overwrite existing config: {target}')
    for track in ('main', 'secondary'):
        config = OutfitterConfig.from_yaml(config_dir / f'outfitter_v48_service_guarantee_{track}.yaml').to_dict()
        config.update(data_dir=str(Path(runtime).resolve()), model='scripted-v49-validation')
        (config_dir / f'outfitter_v49_school_holiday_r2_{track}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config-dir')
    args = parser.parse_args()
    result = build(args.source, args.output)
    if args.config_dir:
        write_configs(args.config_dir, args.output)
    print(json.dumps(result))
