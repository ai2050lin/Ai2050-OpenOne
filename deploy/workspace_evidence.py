"""Search published material and, with archive authorization, verified run artifacts."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import re

from deploy import test_record_catalog as records
from deploy import workspace_service as workspace
from shared.dist_protocol.test_archive import NATIVE_ADDRESS

MAX_ARTIFACT = 16 * 2**20
READ_BUDGET = 64 * 2**20
CASE_BUDGET = 20_000
SOURCES = {'study','run','case','history'}
COVERAGE = {'source','run','case','native','reviewed'}
ADDRESS = re.compile(r'B(\d+)/([HU])(\d+)')


def as_text(value, limit=1000):
    if isinstance(value,str):
        return value[:limit]
    return json.dumps(value,ensure_ascii=False)[:limit] if value is not None else ''


def study_identity(row):
    if row.get('memo') and row.get('phase_sha256') and row.get('phase'):
        return row['memo'],row['phase'],row['phase_sha256']
    return None


def record_row(row, source, groups):
    return {'id':row['id'],'source':source,'phase':row.get('phase'),'title':row['title'],
            'topic':row.get('question') or groups.get(row.get('group')) or row.get('group'),
            'model':row.get('model'),'snapshot':row.get('snapshot'),'status':row.get('status'),
            'method':row.get('type'),'objective':row.get('objective'),'principle':row.get('principle'),
            'result':row.get('result'),'counter':row.get('counter'),'gate':row.get('gate'),
            'run_refs':[item['id'] for item in (row.get('runs') or [])],
            'detail_source':'goodfire' if source=='study' else None,
            'targets':[item for item in row.get('objects',[]) if NATIVE_ADDRESS.fullmatch(str(item))],
            'novelty':row.get('novelty'),'earlier':row.get('earlier'),'note':row.get('case_note'),
            'case_state':row.get('case_state'),'proofs':row.get('proofs'),
            'study_identity':study_identity(row),
            'coverage':['source']}


def lookup_published_asset(identity):
    try:
        root, entry = workspace.entry_for('resources',identity,'public')
        return workspace.asset(root,entry['file'])
    except FileNotFoundError:
        return {}


def observation_refs(visibility):
    """Full run/snapshot/case binding. Read the manifest rather than infer from a shortcut."""
    refs = defaultdict(list)
    archive_entries = workspace.registry('archives')
    for _,entry in archive_entries.values():
        if entry.get('visibility') != visibility:
            continue
        try:
            doc = workspace.manifest(entry['id'],visibility)
        except (FileNotFoundError,ValueError,KeyError,OSError):
            continue
        for row in doc['rows']:
            case = row.get('caseId') or doc.get('caseId') or row.get('doc')
            if case and row['blocks']:
                refs[doc['run'],doc['snapshot'],case].append({
                    'archive':entry['id'],'row':row['key'],'position':row['position'],'token_id':row['tokenId'],
                    'generation_step':doc.get('generationStep'),'blocks':row['blocks'],
                    'trace':doc.get('trace'),
                    'location':entry['location'],'coverage':entry.get('nativeCoverage',[]),
                    'presentation':entry.get('presentation'),'text':row.get('text') or doc.get('text'),
                })
    return refs


def attach_readouts(row, refs, *, trace=None, require_trace=False):
    points = refs.get((row['run_id'],row['snapshot'],row['case_id']),[])
    if require_trace:
        points = [item for item in points if trace and item['trace']==trace]
    if points:
        row['readouts'] = [{k:v for k,v in item.items() if k!='text'} for item in points]
        row['coverage'].append('native')


def published_rows():
    docs = lookup_published_asset('selection-cases')
    verified = {row['id']:row for row in records.annotations()['published_runs']}
    runs = docs.get('evidence',{}).get('runs',[])
    refs = observation_refs('public')
    matched = set()
    topic = '档案选源与记录竞争'
    checked = [run for run in runs if verified.get(run['id'],{}).get('snapshot')==run['snapshot']]
    if checked:
        yield {'source':'study','id':'study-archive-source-selection','title':topic,'topic':topic,
               'run_refs':[run['id'] for run in checked],'coverage':['source'],
               'objective':'查明指定姓名、印刷标签和物理行号时，模型如何选择并读取一条记录。',
               'principle':'复用已发布的三线索选源及记录竞争材料，分开比较整模型答案与单头注意力。',
               'counter':'部分题整模型答对，单头最高注意力落在另一条记录。逐题保留这种分歧。'}
    for run in runs:
        notes = verified.get(run['id'],{})
        same = notes.get('snapshot') == run['snapshot']
        readout = run['readout']
        target = f"B{readout['layer']}/H{readout['head']}"
        basis = {'source':'run','run_id':run['id'],'model':run['model'],'snapshot':run['snapshot'],'topic':topic,
                 'phase':notes.get('phase') if same else None,'targets':[target],'visibility':'public'}
        run_row = {**basis,'id':run['id'],'title':run['title'],'coverage':['source','run','case'],
               'novelty':notes.get('novelty') if same else None,'earlier':notes.get('earlier') if same else None,
               'note':notes.get('missing') if same else None}
        yield run_row
        prompts = docs.get('prompts',{})
        prompt_verified = (prompts.get('run') == run['id'] and prompts.get('snapshot') == run['snapshot']
                           and prompts.get('materialsSha256') == run['provenance'].get('confirm-materials.json'))
        for family in run['families']:
            for sample in family['cases']:
                case = sample['id']
                row = {**basis,'id':f"{run['id']}:{case}",'source':'case','case_id':case,
                       'title':run['title'],'family':str(family.get('family') or family.get('id') or ''),
                       'input':as_text(sample.get('question'),1800),'output':as_text(sample.get('output')),
                       'correct':sample.get('correct'),'coverage':['source','run','case'],
                       'note':'该原题还没有可在中心打开的原生读数。'}
                if isinstance(sample.get('correct'),bool) and sample['correct'] and sample.get('mass'):
                    chosen = max(range(len(sample['mass'])),key=sample['mass'].__getitem__)
                    if chosen != sample.get('gold'):
                        row['counter'] = '模型答对，最高注意力分给了另一条记录。'
                if prompt_verified and case in prompts.get('prompts',{}):
                    row['prompt'] = as_text(prompts['prompts'][case],12_000)
                attach_readouts(row,refs)
                if row.get('readouts'):
                    row['note'] = None
                    matched.add((row['run_id'],row['snapshot'],case))
                    if 'native' not in run_row['coverage']:
                        run_row['coverage'].append('native')
                yield row
    for (run,snapshot,case), points in refs.items():
        if (run,snapshot,case) in matched:
            continue
        row = {'id':f'{run}:{case}','source':'case','run_id':run,'snapshot':snapshot,'case_id':case,
               'visibility':'public','title':case,'input':as_text(points[0].get('text'),1800),
               'coverage':['source','run'],'note':'原场已发布。此入口的逐题评分未单列。'}
        attach_readouts(row,refs)
        yield row


@lru_cache(maxsize=8)
def parse_artifact(path, mtime, size, checksum, json_lines):
    with path.open('rb') as stream:
        payload = stream.read(MAX_ARTIFACT+1)
    if len(payload)!=size or hashlib.sha256(payload).hexdigest()!=checksum:
        raise ValueError('archive artifact checksum mismatch')
    data = [json.loads(line) for line in payload.splitlines() if line.strip()] if json_lines else json.loads(payload)
    if isinstance(data,dict):
        data = data.get('cases') or data.get('questions')
    if not isinstance(data,list):
        raise ValueError('unsupported case artifact format')
    return [row for row in data if isinstance(row,dict)]


def artifact_rows(store, artifact):
    try:
        path = store.download(artifact['sha256'])
        stat = path.stat()
        if stat.st_size != artifact['size'] or stat.st_size>MAX_ARTIFACT:
            return None
        return parse_artifact(path,stat.st_mtime_ns,stat.st_size,artifact['sha256'],artifact['name'].endswith('.jsonl'))
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def artifacts(receipt, role):
    return [item for item in receipt.get('artifacts',[]) if item['role']==role
            and item['availability']=='uploaded' and item['name'].endswith(('.json','.jsonl'))]


def private_rows(omissions, public_cases):
    store = workspace.archive()
    refs = observation_refs('private')
    bytes_left, cases_left = READ_BUDGET, CASE_BUDGET
    for run in store.search_inventory():
        plan, receipt, final = run['plan'], run['receipt'] or {}, run['final'] or {}
        acq = plan['acquisition']
        snapshot, identity = acq['model_snapshot'], run['id']
        fields = [item for item in receipt.get('artifacts',[]) if item['role']=='native_field']
        review = receipt.get('evidence_status')
        verified = run['archive_state']=='server_verified'
        basis = {'run_id':identity,'snapshot':snapshot,'visibility':'private','model':acq['model_id'],
                 'families':plan['family_ids'],'domains':plan['domains'],'topic':'、'.join(plan['family_ids']),
                 'title':plan['title'],'method':plan['analysis']['method_version'],'status':run['state'],
                 'targets':acq['capture'].get('targets',[]),'archive_state':run['archive_state'],
                 'review':review,'coverage':['source','run']}
        if verified and review in {'reviewed','scientifically_verified'}:
            basis['coverage'].append('reviewed')
        run_row = {**basis,'id':identity,'source':'run','objective':plan['objective'],'principle':plan['principle'],
                   'gate':plan['criteria'],'result':as_text(final.get('summary'),2200),
                   'counter':as_text(final.get('counterexamples')),'earlier':run['source_run_id'],
                   'novelty':run['reason'] if run['decision']!='new_acquisition' else None,
                   'note':None if verified else '当前运行尚缺已核验的完整复核包。',
                   'field_notes':[item.get('reason') or item['availability'] for item in fields]}
        found = []
        for obj in artifacts(receipt,'case_results'):
            if bytes_left<obj['size'] or cases_left<=0:
                omissions['budget']+=1
                continue
            bytes_left-=obj['size']
            rows = artifact_rows(store,obj)
            if rows is None:
                omissions['unreadable']+=1
                continue
            found.extend((row,obj['sha256']) for row in rows[:cases_left])
            if len(rows)>cases_left:
                omissions['budget']+=1
            cases_left-=min(cases_left,len(rows))
        questions = {}
        if found:
            for obj in artifacts(receipt,'questions'):
                if bytes_left<obj['size']:
                    continue
                bytes_left-=obj['size']
                for question in artifact_rows(store,obj) or []:
                    key = question.get('case_id') or question.get('id')
                    if isinstance(key,str) and question.get('snapshot',snapshot)==snapshot:
                        questions.setdefault(key,question.get('prompt') or question.get('input') or question.get('text'))
        cases = []
        seen = set()
        for raw, checksum in found:
            case = raw.get('case_id') or raw.get('id')
            if (not isinstance(case,str) or not case or raw.get('run_id',identity)!=identity
                    or raw.get('model_snapshot',snapshot)!=snapshot):
                omissions['identity']+=1
                continue
            arm = as_text(raw.get('arm'),120)
            trace = raw.get('trace_id') if isinstance(raw.get('trace_id'),str) else ''
            unique = (checksum,case,arm,trace)
            if unique in seen:
                continue
            seen.add(unique)
            top = raw.get('output_top')
            row = {**basis,'id':f'{identity}:{checksum}:{case}:{arm}:{trace}', 'source':'case','case_id':case,'arm':arm,
                   'family':raw.get('family_id') or raw.get('family'),'input':as_text(raw.get('input') or
                       raw.get('prompt') or raw.get('question') or raw.get('text') or questions.get(case),12_000),
                   'output':as_text(raw.get('output') or raw.get('actual_output') or (top.get('text') if isinstance(top,dict) else top)),
                   'correct':raw.get('correct'),'counter':as_text(raw.get('counterexample') or raw.get('error')),
                   'coverage':[*basis['coverage'],'case'],'case_sha256':checksum,
                   'note':'该题的逐题文件可读，中心尚无绑定的原生切片。'}
            origin = public_cases.get((raw.get('source_run'),snapshot,case))
            if origin and origin.get('prompt') and origin['prompt']==row['input']:
                row['source_case'] = {'run':origin['run_id'],'case':case}
            attach_readouts(row,refs,trace=trace,require_trace=True)
            if row.get('readouts'):
                row['note'] = None
            cases.append(row)
        if cases:
            run_row['coverage'] = [*basis['coverage'],'case']
        if any(case.get('readouts') for case in cases):
            run_row['coverage'].append('native')
        yield run_row
        yield from cases


def target_matches(row, target, points):
    match = ADDRESS.fullmatch(target)
    if match and points:
        block, kind, index = match.groups()
        return any(block in item['blocks'] and any(
            bound['module']==('attn' if kind=='H' else 'mlp') and
            bound['firstBlock']<=int(block)<=bound['lastBlock'] and bound['firstIndex']<=int(index)<=bound['lastIndex']
            for bound in item['coverage']) for item in points)
    return target in row.get('targets',[]) and not points


def relevance(row, query):
    text = ' '.join(as_text(row.get(key),12_000) for key in ('title','id','phase','topic','method','model','principle',
                   'input','prompt','output','objective','result','counter','gate','family','targets','run_refs','novelty')).casefold()
    if query in text:
        return 3 + int(query in {str(row.get(key,'')).casefold() for key in ('case_id','id','run_id')})
    parts = query.split()
    return 1 if all(part in text for part in parts) else 0


def link_records(rows, pilot_rows):
    known_runs = {row['run_id']:row for row in rows if row['source']=='run'}
    for row in rows:
        if row['source'] not in {'study','history'}:
            continue
        linked = [identity for identity in row['run_refs'] if identity in known_runs
                  and row.get('snapshot') in (None, known_runs[identity].get('snapshot'))]
        if linked:
            row['linked_runs'] = linked
            row['coverage'] += sorted({level for identity in linked for level in known_runs[identity]['coverage']}-{'source'})
            if 'case' in row['coverage'] and row.get('case_state')=='case_link_pending':
                row['note'] = '已找到关联运行的逐题结果，具体原场覆盖见相应运行。'
        if row['source']=='history' and row.get('study_identity') in pilot_rows:
            row['study_id'] = pilot_rows[row['study_identity']]
        row.pop('study_identity',None)


def search(q='', source='', coverage='', snapshot='', family='', target='', position=None, token_id=None,
           counter=False, offset=0, limit=12, authorized=False):
    if source and source not in SOURCES or coverage and coverage not in COVERAGE:
        raise ValueError('unsupported evidence filter')
    if snapshot and not workspace.SHA.fullmatch(snapshot) or target and not NATIVE_ADDRESS.fullmatch(target):
        raise ValueError('invalid snapshot or native address')
    pilot = records.document('goodfire')
    studies = [record_row(row,'study',pilot.get('groups',{})) for row in pilot['records']]
    omissions = Counter()
    rows = [*studies,*published_rows()]
    public_cases = {(row['run_id'],row['snapshot'],row['case_id']):row for row in rows if row['source']=='case'}
    lesson = next((row for row in public_cases.values() if row.get('prompt') and row.get('readouts') and row.get('counter')),None)
    if lesson is None:
        lesson = next((row for row in public_cases.values() if row.get('prompt') and row.get('readouts')),None)
    if authorized:
        rows += [record_row(row,'history',{}) for row in records.catalog('history')]
        rows += list(private_rows(omissions,public_cases))
    by_phase = defaultdict(list)
    for row in pilot['records']:
        by_phase[study_identity(row)].append(row['id'])
    link_records(rows,{key:ids[0] for key,ids in by_phase.items() if key and len(ids)==1})
    totals = Counter(row['source'] for row in rows)
    counts = {key:Counter(row['source'] for row in rows if key in row['coverage']) for key in sorted(COVERAGE)}
    relevant = []
    query = q.casefold().strip()
    for row in rows:
        if (source and row['source']!=source or coverage and coverage not in row['coverage']
                or snapshot and row.get('snapshot')!=snapshot
                or family and family not in [row.get('family'),*row.get('families',[])]
                or counter and not row.get('counter')):
            continue
        points = [item for item in row.get('readouts',[]) if (position is None or item['position']==position)
                  and (token_id is None or item['token_id']==token_id)]
        if (position is not None or token_id is not None) and not points:
            continue
        if target and not target_matches(row,target,points):
            continue
        score = relevance(row,query) if query else 1
        if score:
            relevant.append((score,row,points))
    relevant.sort(key=lambda item:(-item[0], -int(bool(item[2])), -int(item[1]['source']=='study'),item[1]['id']))
    return {'schema':'workspace_evidence_search.v1','visibility':'authorized' if authorized else 'public',
            'scope_totals':totals,'coverage_counts':counts,'omissions':omissions,'total':len(relevant),
            'historical_links':sum(bool(row.get('study_id')) for row in rows if row['source']=='history'),
            'learning_case':{'id':lesson['id'],'case_id':lesson['case_id'],'target':lesson['targets'][0]} if lesson else None,
            'items':[{**row,**({'readouts':points} if points else {})} for _,row,points in relevant[offset:offset+limit]]}
