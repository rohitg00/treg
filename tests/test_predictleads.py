"""PredictLeads on treg's key: the per-record lists hold `limit` rows and refuse a limit outside the
row's range before any hold, every answer settles at the X-Credits-Used header (a billed 4xx
included), and a team's own key keeps PredictLeads' own range unmetered."""
import base64

import httpx
import pytest
from httpx import AsyncClient

from treg import api as A
from treg.config import get_settings
from treg.domain import money
from treg.infra.db import session_maker
from test_marketplace_call import _DropleadsJSONStream, _balance, _entries

CREDIT = 40_000
PLATFORM = 'Basic ' + base64.b64encode(b'PLATFORM-KEY:PLATFORM-TOKEN').decode()
LISTS = {
    'predictleads.jobs.discover': ({}, 30, 1),
    'predictleads.technologies.users': ({'technology_id_or_fuzzy_name': 'hubspot'}, 30, 1),
    'predictleads.technologies.list': ({}, 100, 3),
    'predictleads.companies.similar': ({'company_id_or_domain': 'example.com'}, 20, 1),
}


@pytest.fixture
def predictleads_on(monkeypatch):
    monkeypatch.setenv('TREG_PLATFORM_KEY_PREDICTLEADS', 'PLATFORM-KEY:PLATFORM-TOKEN')
    monkeypatch.setenv('TREG_PLATFORM_PROVIDERS', 'predictleads')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _fund(clients):
    org_id = (await clients.get('/orgs')).json()[0]['org_id']
    async with session_maker() as db:
        await money.grant(db, org_id, amount_micro=20_000_000, kind='promotional', once=False)
        await db.commit()


async def _call(clients, monkeypatch, endpoint, params, answer, seen=None):
    def upstream(request):
        if seen is not None:
            seen.append(request)
        status, doc, credits = answer
        headers = {'content-type': 'application/json'}
        if credits is not None:
            headers['x-credits-used'] = credits
        return httpx.Response(status, headers=headers, stream=_DropleadsJSONStream(doc))

    async with AsyncClient(transport=httpx.MockTransport(upstream)) as vendor:
        monkeypatch.setattr(A.app.state, 'http', vendor)
        return await clients.get(f'/call/{endpoint}', params=params)


async def _reserves(clients):
    return [-e['amount_micro'] for e in await _entries(clients) if e['kind'] == 'reserve']


@pytest.mark.parametrize('endpoint', sorted(LISTS))
async def test_a_list_holds_its_limit_and_settles_the_reported_credits(clients, predictleads_on, monkeypatch, endpoint):
    path, _, credits = LISTS[endpoint]
    await _fund(clients)
    seen = []
    before = await _balance(clients)
    r = await _call(clients, monkeypatch, endpoint, {**path, 'limit': '10'},
                    (200, {'data': [{}, {}]}, str(2 * credits)), seen)
    assert r.status_code == 200, r.text
    assert seen[0].headers['authorization'] == PLATFORM
    assert seen[0].url.params.get_list('limit') == ['10']
    assert await _reserves(clients) == [money.with_margin(10 * credits * CREDIT)]
    assert before - await _balance(clients) == money.with_margin(2 * credits * CREDIT)


@pytest.mark.parametrize('endpoint', sorted(LISTS))
async def test_a_list_without_a_limit_holds_the_documented_default(clients, predictleads_on, monkeypatch, endpoint):
    path, default, credits = LISTS[endpoint]
    await _fund(clients)
    seen = []
    r = await _call(clients, monkeypatch, endpoint, path, (200, {'data': []}, '0'), seen)
    assert r.status_code == 200, r.text
    assert 'limit' not in seen[0].url.params
    assert await _reserves(clients) == [money.with_margin(default * credits * CREDIT)]


@pytest.mark.parametrize('endpoint', sorted(LISTS))
@pytest.mark.parametrize('items', [
    [('limit', '0')], [('limit', '101')], [('limit', '1000')], [('limit', '1001')], [('limit', 'abc')],
    [('limit', '1e3')], [('limit', ' 50')], [('limit', '')], [('limit', '-5')], [('limit', '5.0')],
    [('limit', '5'), ('limit', '1000')], [('limit', '5'), ('limit', '5')],
])
async def test_a_list_refuses_a_limit_outside_its_range_before_any_hold(clients, predictleads_on, monkeypatch, endpoint, items):
    path, _, _ = LISTS[endpoint]
    seen = []
    before = await _balance(clients)
    r = await _call(clients, monkeypatch, endpoint, list(path.items()) + items, (200, {'data': []}, '0'), seen)
    assert r.status_code == 400, r.text
    assert r.json()['detail']['parameter'] in ('query.limit', 'limit')
    assert not seen
    assert await _balance(clients) == before
    assert not [e for e in await _entries(clients) if e['kind'] in ('reserve', 'settle', 'release')]


async def test_similar_stops_at_its_vendor_max_of_50(clients, predictleads_on, monkeypatch):
    await _fund(clients)
    path = {'company_id_or_domain': 'example.com'}
    ok = await _call(clients, monkeypatch, 'predictleads.companies.similar', {**path, 'limit': '50'},
                     (200, {'data': []}, '0'))
    assert ok.status_code == 200, ok.text
    over = await _call(clients, monkeypatch, 'predictleads.companies.similar', {**path, 'limit': '51'},
                       (200, {'data': []}, '0'))
    assert over.status_code == 400, over.text


@pytest.mark.parametrize('answer,charged', [
    ((200, {'data': [{}]}, '1'), CREDIT),
    ((202, {'data': [], 'meta': {'message_type': 'accepted'}}, '1'), CREDIT),
    ((200, {'data': []}, '1'), CREDIT),
    ((400, {'error': {'type': 'bad_request', 'message': 'invalid domain'}}, '1'), CREDIT),
    ((404, {'error': {'type': 'not_found', 'message': 'not found'}}, '1'), CREDIT),
    ((422, {'error': {'type': 'unprocessable', 'message': 'bad'}}, '1'), CREDIT),
    ((401, {'error': {'type': 'unauthorized', 'message': 'Authentication failed.'}}, None), 0),
    ((429, {'error': {'type': 'rate_limited', 'message': 'slow down'}}, None), 0),
    ((500, {'error': {'type': 'internal', 'message': 'boom'}}, None), 0),
    ((400, {'error': {'type': 'bad_request', 'message': 'invalid domain'}}, '0'), 0),
])
async def test_a_per_response_row_settles_the_header_and_a_billed_4xx_passes_through(clients, predictleads_on, monkeypatch, answer, charged):
    await _fund(clients)
    before = await _balance(clients)
    r = await _call(clients, monkeypatch, 'predictleads.companies.job_openings',
                    {'company_id_or_domain': 'example.com', 'limit': '1000'}, answer)
    assert r.status_code == answer[0], r.text
    assert before - await _balance(clients) == money.with_margin(charged)


async def test_technology_detail_holds_and_settles_three_credits(clients, predictleads_on, monkeypatch):
    await _fund(clients)
    before = await _balance(clients)
    r = await _call(clients, monkeypatch, 'predictleads.technologies.detail',
                    {'id_or_fuzzy_name': 'hubspot'}, (200, {'data': [{}]}, '3'))
    assert r.status_code == 200, r.text
    assert await _reserves(clients) == [money.with_margin(3 * CREDIT)]
    assert before - await _balance(clients) == money.with_margin(3 * CREDIT)


async def test_an_answer_without_the_header_settles_at_the_hold(clients, predictleads_on, monkeypatch):
    await _fund(clients)
    before = await _balance(clients)
    r = await _call(clients, monkeypatch, 'predictleads.jobs.discover', {'limit': '4'},
                    (200, {'data': [{}, {}, {}, {}]}, None))
    assert r.status_code == 200, r.text
    assert before - await _balance(clients) == money.with_margin(4 * CREDIT)


@pytest.mark.parametrize('items,status', [
    ([('limit', '101')], 200), ([('limit', '1000')], 200), ([], 200), ([('limit', '1001')], 400),
])
async def test_own_key_keeps_predictleads_range_and_is_unmetered(clients, predictleads_on, monkeypatch, items, status):
    saved = await clients.post('/secrets', json={'name': 'predictleads', 'value': 'OWN-KEY:OWN-TOKEN'})
    assert saved.status_code == 200, saved.text
    seen = []
    before = await _balance(clients)
    r = await _call(clients, monkeypatch, 'predictleads.jobs.discover', items, (200, {'data': []}, '1000'), seen)
    assert r.status_code == status, r.text
    if status == 200:
        assert seen[0].headers['authorization'] != PLATFORM
        assert seen[0].url.params.get_list('limit') == [v for _, v in items]
        assert 'x-treg-cost-micro' not in r.headers
    else:
        assert not seen
    assert await _balance(clients) == before
