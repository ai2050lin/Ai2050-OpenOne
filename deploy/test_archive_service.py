"""Authenticated private test archive. No GPU execution or public evidence publication."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from deploy.research_library import services
from deploy import test_record_catalog
from deploy.storage_policy import StorageBlocked
from shared.dist_protocol.test_archive import Archive, CHUNK_BYTES, META_BYTES, RUN_FILTERS, Conflict

router = APIRouter(prefix='/api/test-archive', tags=['private-test-archive'])


def actor(request):
    D = services()
    token = request.headers.get('x-admin-token', '')
    if token:
        D._auth_admin(token)
        return 'center-admin'
    user = D._auth_call(D._accounts.require_admin, request.headers.get('authorization', ''))
    return 'account:' + user['user_id']


def archive():
    return Archive(Path(services()._DATA_DIR) / 'research_library' / 'test_archive')


def call(action):
    try:
        return action()
    except StorageBlocked as exc:
        raise HTTPException(507, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except sqlite3.OperationalError as exc:
        raise HTTPException(503, 'archive temporarily busy; retry without discarding local files') from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


async def body(request, limit=META_BYTES, binary=False):
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > limit:
            raise HTTPException(413, 'request body exceeds endpoint limit')
        raw.extend(chunk)
    if binary:
        return bytes(raw)
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, 'invalid JSON') from exc
    if not isinstance(result, dict):
        raise HTTPException(422, 'JSON object required')
    return result


@router.get('/capabilities')
def capabilities():
    return {'schema': 'research_test_archive_api.v1', 'private': True,
            'authentication': 'existing admin session or X-Admin-Token',
            'chunk_bytes': CHUNK_BYTES, 'metadata_bytes': META_BYTES,
            'workflow': ['preflight', 'runs', 'uploads', 'finalize', 'receipt'],
            'run_filters': RUN_FILTERS,
            'historical_coverage': 'registered_records_only', 'runs_models': False}


@router.post('/preflight')
async def preflight(request: Request):
    who = actor(request)
    doc = await body(request)
    return await run_in_threadpool(call, lambda: archive().preflight(doc, who, test_record_catalog.similar))


@router.get('/preflights/{preflight_id}')
def read_preflight(preflight_id: str, request: Request):
    actor(request)
    return call(lambda: archive().preflight_receipt(preflight_id))


@router.get('/history')
def history(request: Request, q: str = Query('', max_length=1000),
            offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    actor(request)
    return call(lambda: test_record_catalog.search(q, 'history', offset, limit))


@router.get('/history/{record_id}')
def historical_record(record_id: str, request: Request):
    actor(request)
    return call(lambda: test_record_catalog.detail(record_id))


@router.post('/runs')
async def start(request: Request):
    who = actor(request)
    doc = await body(request)
    return await run_in_threadpool(call, lambda: archive().start(doc, who))


@router.get('/runs')
def query(request: Request, q: str = Query('', max_length=500), domain: str = '', model: str = '',
          state: str = '', family: str = '', snapshot: str = Query('', max_length=64),
          target: str = Query('', max_length=20), method: str = Query('', max_length=200),
          archive_state: str = Query('', max_length=40), artifact_role: str = Query('', max_length=40),
          decision: str = Query('', max_length=40), limit: int = Query(50, ge=1, le=200),
          offset: int = Query(0, ge=0)):
    actor(request)
    return call(lambda: archive().query(q=q, domain=domain, model=model, state=state,
                                       family=family, snapshot=snapshot, target=target, method=method,
                                       archive_state=archive_state, artifact_role=artifact_role, decision=decision,
                                       limit=limit, offset=offset))


@router.get('/runs/{run_id}')
def detail(run_id: str, request: Request):
    actor(request)
    return call(lambda: archive().get(run_id))


@router.post('/runs/{run_id}/finalize')
async def finalize(run_id: str, request: Request):
    who = actor(request)
    doc = await body(request)
    return await run_in_threadpool(call, lambda: archive().finalize(run_id, doc, who))


@router.get('/runs/{run_id}/receipt')
def receipt(run_id: str, request: Request, verify: bool = True):
    actor(request)
    return call(lambda: archive().receipt(run_id, verify))


@router.post('/runs/{run_id}/revisions')
async def recovery(run_id: str, request: Request):
    who = actor(request)
    doc = await body(request)
    return await run_in_threadpool(call, lambda: archive().finalize(
        run_id, doc['final'], who, doc['parent_manifest_sha256']))


@router.get('/runs/{run_id}/revisions')
def revisions(run_id: str, request: Request):
    actor(request)
    return call(lambda: archive().revisions(run_id))


@router.post('/uploads')
async def upload_init(request: Request):
    who = actor(request)
    doc = await body(request)
    return await run_in_threadpool(call, lambda: archive().upload_init(doc, who))


@router.get('/uploads/{upload_id}')
def upload_status(upload_id: str, request: Request):
    who = actor(request)
    return call(lambda: archive().upload_status(upload_id, who))


@router.put('/uploads/{upload_id}')
async def upload_chunk(upload_id: str, request: Request, offset: int = Query(..., ge=0)):
    who = actor(request)
    raw = await body(request, CHUNK_BYTES, binary=True)
    return await run_in_threadpool(call, lambda: archive().upload_chunk(upload_id, offset, raw, who))


@router.post('/uploads/{upload_id}/finish')
def upload_finish(upload_id: str, request: Request):
    who = actor(request)
    return call(lambda: archive().upload_finish(upload_id, who))


@router.get('/objects/{sha256}')
def download(sha256: str, request: Request):
    actor(request)
    path = call(lambda: archive().download(sha256))
    return FileResponse(path, media_type='application/octet-stream', filename=sha256)
