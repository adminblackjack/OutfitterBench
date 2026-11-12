"""Actor-visible query contracts match execution, including empty placeholders."""
import copy
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from outfitter.config import OutfitterConfig
from outfitter.data import OutfitterDataset
from outfitter.environment import OutfitterEnvironment
from outfitter.tool_executor import ToolExecutor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['rental_only', 'rental_and_retail'])
def runtime(request):
    config = replace(OutfitterConfig.from_yaml(ROOT/'configs/outfitter_v54_main.example.yaml'),
                     assortment_mode=request.param)
    env = OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))
    return env.reset(), ToolExecutor(env)


def parameters(tools, name):
    return next(d['function']['parameters'] for d in tools.definitions() if d['function']['name'] == name)


@pytest.mark.parametrize('placeholder', ['', '  ', '\t\n', None])
def test_empty_order_id_is_list_mode_and_read_only(runtime, placeholder):
    state, tools = runtime
    expected = tools.execute(state, 'purchase_order_status', {'status': 'in_transit', 'limit': 20})
    args = {'order_id': placeholder, 'status': 'in_transit', 'limit': 20}
    before = copy.deepcopy(state.to_dict()); original_args = copy.deepcopy(args)
    jsonschema.validate(args, parameters(tools, 'purchase_order_status'))
    actual = tools.execute(state, 'purchase_order_status', args)
    assert actual.ok and actual.data == expected.data
    assert state.to_dict() == before and args == original_args


@pytest.mark.parametrize('field,value', [('status', 'in_transit'), ('placed_day_from', 1),
    ('placed_day_to', 3), ('limit', 20)])
def test_real_order_id_cannot_mix_filters(runtime, field, value):
    state, tools = runtime
    args = {'order_id': 'po-00001', field: value}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(args, parameters(tools, 'purchase_order_status'))
    before = copy.deepcopy(state.to_dict())
    actual = tools.execute(state, 'purchase_order_status', args)
    assert not actual.ok and 'only argument' in actual.data['error']
    assert 'omit order_id' in actual.data['error']
    assert state.to_dict() == before


def test_exact_order_and_list_results_stay_consistent(runtime):
    state, tools = runtime
    offer = tools.environment.offers[state.assets[0].offer_id]
    result = tools.execute(state, 'place_purchase_order', {'seller_id': offer.seller_id,
        'items': [{'offer_id': offer.offer_id, 'packs': 1}]})
    assert result.ok
    order_id = result.data['order']['order_id']
    args = {'order_id': order_id}
    jsonschema.validate(args, parameters(tools, 'purchase_order_status'))
    before = copy.deepcopy(state.to_dict())
    assert tools.execute(state, 'purchase_order_status', args).data['order']['order_id'] == order_id
    missing = tools.execute(state, 'purchase_order_status', {'order_id': 'does-not-exist'})
    assert not missing.ok and 'unknown order_id' in missing.data['error']
    assert state.to_dict() == before


@pytest.mark.parametrize('placeholder', ['', '  ', '\t\n', None])
def test_index_accepts_empty_query_and_legacy_pagination(runtime, placeholder):
    state, tools = runtime
    args = {'view': 'index', 'query': placeholder, 'cursor': 0, 'limit': 20}
    original = copy.deepcopy(args); before = copy.deepcopy(state.to_dict())
    jsonschema.validate(args, parameters(tools, 'search_catalog'))
    actual = tools.execute(state, 'search_catalog', args)
    expected = tools.execute(state, 'search_catalog', {'view': 'index'})
    assert actual.ok and actual.data == expected.data
    assert args == original and state.to_dict() == before


def test_index_rejects_nonempty_search_in_schema_and_runtime(runtime):
    state, tools = runtime
    args = {'view': 'index', 'query': 'sleeping bag'}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(args, parameters(tools, 'search_catalog'))
    result = tools.execute(state, 'search_catalog', args)
    assert not result.ok and 'view="offers"' in result.data['error']


@pytest.mark.parametrize('args,expected_view', [({}, 'index'), ({'query': ''}, 'offers'),
    ({'query': None}, 'offers'), ({'query': 'sleeping bag'}, 'offers'),
    ({'view': 'offers', 'query': None}, 'offers')])
def test_catalog_default_view_and_empty_search(runtime, args, expected_view):
    state, tools = runtime
    jsonschema.validate(args, parameters(tools, 'search_catalog'))
    result = tools.execute(state, 'search_catalog', args)
    assert result.ok and result.data['view'] == expected_view


@pytest.mark.parametrize('name,key', [('search_catalog', 'query'), ('purchase_order_status', 'order_id')])
@pytest.mark.parametrize('value', [False, 0, [], {}])
def test_wrong_optional_string_types_not_silently_ignored(runtime, name, key, value):
    state, tools = runtime
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({key: value}, parameters(tools, name))
    result = tools.execute(state, name, {key: value})
    assert not result.ok and 'string or null' in result.data['error']


def test_final_actor_descriptions_survive_overrides(runtime):
    _, tools = runtime
    descriptions = {d['function']['name']: d['function']['description'] for d in tools.definitions()}
    assert 'only order_id' in descriptions['purchase_order_status']
    assert 'whitespace-only' in descriptions['purchase_order_status']
    assert 'view=index' in descriptions['search_catalog']
    assert 'query must be omitted, null or blank' in descriptions['search_catalog']


def test_legacy_query_semantics_unchanged():
    from release_fixtures import v50_runtime
    config = replace(OutfitterConfig.from_yaml(ROOT/'configs/outfitter_v54_main.example.yaml'),
                     data_dir=str(v50_runtime()))
    env = OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))
    state, tools = env.reset(), ToolExecutor(env)
    assert not tools.execute(state, 'search_catalog', {'view': 'index', 'query': ''}).ok
    assert not tools.execute(state, 'purchase_order_status', {'order_id': '', 'limit': 20}).ok
