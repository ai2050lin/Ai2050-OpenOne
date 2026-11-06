"""Owner-scoped, expiring node controls. Nodes grant a fresh session locally.

Only pause/resume/stop are accepted, never shell, file paths or model downloads.
Local pause/stop revoke the grant; remote control cannot restart a stopped node.
"""
import json
import time
import uuid
from fastapi import APIRouter, Header, HTTPException

router=APIRouter(prefix='/api/research-nodes',tags=['node-control'])


def service():
    from .research_library import services
    return services()


def db():
    D=service();conn=D._db()
    with D._db_lock:
        conn.executescript('''CREATE TABLE IF NOT EXISTS node_control_state(node_id TEXT PRIMARY KEY,
            grant_id TEXT NOT NULL,state TEXT NOT NULL,updated REAL NOT NULL,limits_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS node_commands(id TEXT PRIMARY KEY,node_id TEXT NOT NULL,
            actor TEXT NOT NULL,grant_id TEXT NOT NULL,action TEXT NOT NULL,created REAL NOT NULL,
            expires REAL NOT NULL,status TEXT NOT NULL);''')
    return conn


@router.get('')
def nodes(authorization:str=Header(default='')):
    D=service();user=D._auth_call(D._accounts.require_user,authorization);conn=db()
    rows=conn.execute('SELECT node_id,name,last_seen FROM nodes WHERE owner_user_id=?',(user['user_id'],)).fetchall()
    result=[]
    for row in rows:
        status=conn.execute('SELECT * FROM node_control_state WHERE node_id=?',(row['node_id'],)).fetchone()
        history=[dict(r) for r in conn.execute('SELECT id,action,created,status FROM node_commands WHERE node_id=? ORDER BY created DESC LIMIT 10',(row['node_id'],))]
        result.append({**dict(row),'control':dict(status) if status else None,'history':history})
    return {'nodes':result}


@router.post('/{node_id}/commands')
def command(node_id:str,body:dict,authorization:str=Header(default='')):
    D=service();user=D._auth_call(D._accounts.require_user,authorization);conn=db()
    if set(body)!={'action'} or body['action'] not in ('pause','resume','stop'):raise HTTPException(422,'只允许 pause / resume / stop')
    with D._db_lock:
        owner=conn.execute('SELECT owner_user_id FROM nodes WHERE node_id=?',(node_id,)).fetchone()
        if not owner or owner['owner_user_id']!=user['user_id']:raise HTTPException(403,'只能控制本人绑定的节点')
        state=conn.execute('SELECT * FROM node_control_state WHERE node_id=?',(node_id,)).fetchone()
        if not state or not state['grant_id'] or state['updated']<time.time()-30:raise HTTPException(409,'节点未授权托管或已离线；请在本机开启')
        cid=uuid.uuid4().hex;now=time.time()
        conn.execute('INSERT INTO node_commands VALUES(?,?,?,?,?,?,?,?)',(cid,node_id,user['user_id'],state['grant_id'],body['action'],now,now+120,'queued'))
        conn.commit()
    return {'id':cid,'status':'queued','expires_in':120}


@router.post('/poll')
def poll(body:dict,x_node_token:str=Header(default='')):
    D=service();node=D._auth_node(x_node_token);conn=db()
    grant=body.get('grant_id','');state=body.get('state','')
    if not isinstance(grant,str) or len(grant)>80 or state not in ('stopped','paused','running','waiting','error'):raise HTTPException(422,'Invalid node state')
    ack=body.get('ack',[])
    if not isinstance(ack,list) or len(ack)>50 or not all(isinstance(i,str) for i in ack):raise HTTPException(422,'Invalid acknowledgements')
    limits=body.get('limits',{})
    if not isinstance(limits,dict) or len(json.dumps(limits))>2048:raise HTTPException(422,'Invalid limits')
    # Anonymous participants cannot create remote-control grants.
    if not node['owner_user_id']:grant=''
    with D._db_lock:
        now=time.time()
        conn.execute('INSERT OR REPLACE INTO node_control_state VALUES(?,?,?,?,?)',(node['node_id'],grant,state,now,json.dumps(limits)))
        for cid in ack:conn.execute("UPDATE node_commands SET status='applied' WHERE id=? AND node_id=? AND status IN ('queued','delivered')",(cid,node['node_id']))
        conn.execute("UPDATE node_commands SET status='expired' WHERE node_id=? AND expires<? AND status IN ('queued','delivered')",(node['node_id'],now))
        rows=conn.execute("SELECT id,action,grant_id FROM node_commands WHERE node_id=? AND grant_id=? AND expires>=? AND status IN ('queued','delivered') ORDER BY created LIMIT 20",(node['node_id'],grant,now)).fetchall() if grant else []
        for row in rows:conn.execute("UPDATE node_commands SET status='delivered' WHERE id=?",(row['id'],))
        conn.commit()
    return {'commands':[dict(r) for r in rows]}
