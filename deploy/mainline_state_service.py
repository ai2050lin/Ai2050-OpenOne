"""Public immutable research status; reviewed publication uses existing admin auth."""
from fastapi import APIRouter, Header, HTTPException, Response

from shared.dist_protocol.mainline_state import MainlineStore

router = APIRouter(prefix='/api/research/mainline', tags=['research-mainline'])


def store():
    from deploy.research_library import library
    return MainlineStore(library())


@router.get('')
def snapshot(response: Response):
    try:
        result = store().ensure()
    except (ValueError, OSError) as exc:
        raise HTTPException(503, '研究状态记录不可用：' + str(exc)) from exc
    response.headers['Cache-Control'] = 'no-store'
    response.headers['ETag'] = '"' + result['revision'] + '"'
    return result


@router.post('')
def publish(bundle: dict, x_admin_token: str = Header(default=''), authorization: str = Header(default='')):
    from deploy.research_library import _registry_admin
    _registry_admin(x_admin_token, authorization)
    try:
        store().ensure()
        return store().publish(bundle)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
