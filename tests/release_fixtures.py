"""Legacy rule fixtures combined with the currently shipped public weather.

These synthetic regression fixtures are not frozen historical experiment data.
Only their weather provenance/hashes are rebased when the distribution changes;
production loaders retain every integrity check.
"""
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import json

_temporary = {}


@lru_cache(maxsize=6)
def _legacy(version):
    root = Path(__file__).resolve().parents[1]
    temporary = tempfile.TemporaryDirectory(prefix=f'outfitter-v{version}-regression-')
    _temporary[version] = temporary
    target = Path(temporary.name)/'runtime'
    shutil.copytree(root/'src/outfitter/runtime_data', target)
    for path in (root/'tests/fixtures/v52_product_payloads').iterdir():
        shutil.copy2(path, target/path.name)
    shutil.copy2(root/f'tests/fixtures/v{version}_metadata.json', target/'metadata.json')
    shutil.copy2(root/'tests/fixtures/v50_initial_assets.jsonl', target/'initial_assets.jsonl')
    if version in (47,48):
        shutil.copy2(root/'tests/fixtures/v48_market_rules.json', target/'market_rules.json')
    current = json.loads((root/'src/outfitter/runtime_data/metadata.json').read_text())
    path = target/'metadata.json'
    legacy = json.loads(path.read_text())
    for key in ('weather_dataset_id', 'weather_source', 'weather_source_year', 'weather_build'):
        legacy[key] = current[key]
    legacy['content_sha256']['weather.jsonl'] = current['content_sha256']['weather.jsonl']
    legacy['content_sha256'].pop('weather_finalinfo.pkl', None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    return target


def v47_runtime():
    return _legacy(47)


def v48_runtime():
    return _legacy(48)


def v49_runtime():
    return _legacy(49)


def v50_runtime():
    return _legacy(50)


def v51_runtime():
    return _legacy(51)


def v52_runtime():
    return _legacy(52)
