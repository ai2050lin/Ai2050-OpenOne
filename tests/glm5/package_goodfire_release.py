"""Build the Goodfire production research increment from hashed, opened runs."""

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.research_ingest_bundle import verified_objects
from deploy.storage_policy import check_write, staging_file

SOURCE = ROOT / 'tests/glm5/result'
OLD = ROOT / 'release/deployment-20261117'
OUT = ROOT / 'release/deployment-20261118-goodfire'
CHECK = SOURCE / 'goodfire_release_20261118/validation.json'
CENTER = 'ai2050-center-1.0.40.tar.gz'
WEB = 'ai2050-web-1.0.40.zip'
UPLOAD = 'goodfire-phases-2970-2976-research-upload.tar.gz'
ARCHIVE = OUT.parent / 'AI2050-Goodfire-deployment-20261118.zip'
BASE_FILES = (CENTER, CENTER + '.sha256', WEB, 'upgrade_center.sh', 'research_ingest_bundle.py')
DOCS = ('README.md', 'GOODFIRE_REVIEW.md', 'upgrade_goodfire.py')

# Only opened measurements are eligible for public indexing. Frozen plans that
# name sealed materials are represented by their hashes and safe identity fields.
STUDIES = (
    (2970, 'four_20261117', 'plan', 'confirm-summary',
     ('candidate', 'retrospective', 'redundancy-diagnostic'), ('behavior', 'forecast', 'fusion', 'fusion_counterexamples', 'causal', 'path'),
     '四类范式组合与随机单元基线',
     '128 个新家族的组合预测。旧面板的 25 套同层随机组中 24 套重建误差更低，候选组有效秩更低。',
     [('b30-u12591', 30, 'mlp', {'unit': 12591}, 'mlp.gate_up', 'observation', 'B30/U12591 条件响应与原生权重候选')],
     ['57c765c3b3fb61d4de66-1111']),
    (2972, 'path_20261117', 'plan', 'analysis',
     ('behavior', 'legacy-plan', 'legacy-replay'), ('behavior', 'causal', 'fusion'),
     '逐头阻断校准与 1011／1111 条件差异',
     '重放修正六个家族的空操作失配，按实际 BF16 剂量匹配。1011 二阶项改善，1111 在 125/128 家族有害。候选头至 G/U 的链条未确认。',
     [], ['57c765c3b3fb61d4de66-1111']),
    (2973, 'archive_20261118', 'design', 'analysis',
     ('behavior', 'candidate', 'holdout-plan'), ('archives', 'behavior', 'prediction', 'native'),
     '六套等长档案和条件化二阶预测',
     '128 个新家族上，只在 1011 加二阶项的总体 KL 为 0.07124，两条基线为 0.20657 和 0.48918。原生来源及 G/U 恢复均未过门。',
     [], ['bb47d7005ef8df98b644-1111-r2-111']),
    (2974, 'position_20261118', 'design', 'analysis',
     ('behavior', 'candidate', 'holdout-plan'), ('gate', 'location', 'fusion', 'native'),
     '等词面档案与请求换位、早层可读性',
     '1111 的档案作用随位置由正转负，再随请求后移转正。早层可读，三位置联合阻断未通过。',
     [('b0-archive', 0, 'residual', {}, 'attention.post_residual', 'observation', 'B0 注意力后出现档案敏感残差信号'),
      ('b24-triple', 24, 'residual', {}, 'block.post', 'negative', 'B24 三位置联合阻断未过门')],
     ['423a2ccacc74fa7ad5f1-1011-l0-a1', '06ed17da2f3e373f72e0-1111-l0-a1']),
    (2975, 'order_20261118', 'design', 'analysis',
     ('behavior', 'candidate', 'holdout-plan', 'marker-plan', 'marker-native-plan'),
     ('gate', 'marker', 'marker_native_difference', 'native', 'old_marker_native', 'old_marker_native_gate'),
     '1111 交叉换位与整组标记复测',
     '同家族换两处标记后反转由 116/128 降至 60/128；较窄路径在旧、新标记上均未通过。',
     [], ['c769e56de59653715d4b-1111-l2-a1']),
    (2976, 'marker_20261118', 'design', 'analysis',
     ('behavior', 'candidate', 'holdout-plan'), ('behavior', 'factorial', 'native'),
     '独立占位因素及请求后移的原生联合干预',
     '128 家族中，单换档案占位的反转由 123/128 降至 84/128，单换请求仍为 123/128。64 家族原生验证中，四段联合削弱两标记组的请求后移效应 39.8%／34.0%，全词表首选 token 未翻转。档案后移和特选 G/U 未过门。',
     [('four-stage-request', 31, 'residual', {'blocks': [0, 8, 20, 31], 'counts': [4, 8, 8, 82]},
       'blocks.pre', 'intervention', 'B0/B8/B20/B31 四段联合 · 请求后移效应'),
      ('four-stage-archive', 31, 'residual', {'blocks': [0, 8, 20, 31], 'counts': [4, 8, 8, 82]},
       'blocks.pre', 'negative', '同组路径 · 档案后移未过门'),
      ('b32-receiver', 32, 'mlp', {}, 'mlp.gate_up', 'negative', 'B32 所选 G/U · 恢复未超过随机基线')],
     ['m0-4328e1f20a120e52a8fe-1111-l2-a1', 'm0-d5779493e7548057001f-1111-l3-a1']),
)


def canonical(doc):
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    target = check_write(path, len(raw))
    target.parent.mkdir(parents=True, exist_ok=True)
    with staging_file(target) as partial:
        partial.write_bytes(raw)


def study_records(spec, records):
    phase, folder, design, result, extra, keys, label, summary, specs, examples = spec
    source, = SOURCE.glob(f'paradigm_{folder}_v1')
    manifest_raw = (source / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    file_names = [f'{name}.json' for name in (design, result, *extra)] + ['report.md']
    for name in file_names:
        raw = (source / name).read_bytes()
        if dict(bytes=len(raw), sha256=sha(raw)) != manifest['files'][name]:
            raise ValueError(f'Run manifest mismatch: {source.name}/{name}')
    plan = json.loads((source / f'{design}.json').read_bytes())
    model = plan.get('snapshot') or plan['model']
    snapshot = model['snapshot']
    run = plan['run']
    if manifest.get('run', run) != run:
        raise ValueError(f'Run identity mismatch: {run}')
    measurements = json.loads((source / f'{result}.json').read_bytes())
    material = plan.get('material') or plan['materials']
    splits = {split: {k: v for k, v in fields.items() if k in ('families', 'sha', 'identity', 'views')}
              for split, fields in material.items() if isinstance(fields, dict) and 'families' in fields}
    provenance = dict(run_manifest_sha256=sha(manifest_raw), frozen_at=plan['at'],
                      source_file_sha256={k: manifest['files'][k]['sha256'] for k in file_names},
                      runner_source_sha256=manifest['sources'], model_file_identity=model['files'],
                      splits=splits)

    def add(schema, **fields):
        raw = canonical(dict(schema=schema, kind='real', run_id=run, model_snapshot=snapshot, phase=phase, **fields))
        digest = sha(raw)
        records[digest] = raw
        return digest

    title = f'Phase {phase} · Qwen3-14B · {label}'
    aux = {'retrospective': json.loads((source / 'retrospective.json').read_bytes()),
           'redundancy_diagnostic': json.loads((source / 'redundancy-diagnostic.json').read_bytes())} if phase == 2970 else {}
    evidence = add('goodfire_opened_measurements.v1',
                   measurements={key: measurements[key] for key in keys}, auxiliary=aux, provenance=provenance)
    report = add('goodfire_paradigm_report.v1', title=title, summary=summary,
                 scope='本地原权重 BF16；受控档案计数与格式选择；独立设备复算待完成。',
                 counterexamples=examples, evidence_sha256=evidence, provenance=provenance)
    cards = [dict(id=f'{run}:{name}', run_id=run, label=card_title, layer=layer, boundary=boundary,
                  index=dict(space=space, **index), method='原生 BF16 固定候选及家族对照',
                  scope=dict(dataset='Goodfire · 四类范式', phase=phase, summary=summary),
                  evidence=dict(level=level, artifact_sha256=evidence), counterexamples=examples)
             for name, layer, space, index, boundary, level, card_title in specs]
    graph = add('research_graph.v1', feature_cards=cards, graph_edges=[])
    binding = add('research_space_binding.v1', report_sha256=report, graph_sha256=graph, traces=[],
                  context=dict(study_type='goodfire_paradigm', model_dimensions=dict(n_layers=40, d_model=5120, d_ff=17408, n_heads=40)))
    stages = dict(registration_validated=dict(status='done', report_sha256=report, graph_sha256=graph),
                  heldout=dict(status='done', scope='冻结后的家族留出，联合锁箱留存于研究侧'),
                  native_intervention=dict(status='done', scope=summary),
                  independent_recomputation=dict(status='pending'))
    work = add('research_workpackage.v1', title=title, stages=stages)
    return dict(run_id=run, model_snapshot=snapshot, report_sha256=report, graph_sha256=graph,
                binding_sha256=binding, workpackage_sha256=work, evidence_sha256=evidence), source


def build_data():
    check_write(OUT, 192 * 2**20, operation='Goodfire 发布及隔离验收峰值')
    records = {}
    bundles = [study_records(spec, records) for spec in STUDIES]
    manifest = dict(schema='ai2050_research_upload.v1', studies=[b[0] for b in bundles],
                    objects=[dict(sha256=digest) for digest in records])
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        members = {'MANIFEST.json': canonical(manifest)} | {f'objects/{key}.json': raw for key, raw in records.items()}
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    data.seek(0)
    with tarfile.open(fileobj=data, mode='r:gz') as archive:
        if dict(verified_objects(archive, manifest)) != records:
            raise ValueError('Data archive did not read back identically')
    write(OUT / UPLOAD, data.getvalue())
    old_files = json.loads((OLD / 'RELEASE_MANIFEST.json').read_bytes())['files']
    for name in BASE_FILES:
        raw = (OLD / name).read_bytes()
        if sha(raw) != old_files[name]['sha256']:
            raise ValueError(f'Prior release byte identity mismatch: {name}')
        write(OUT / name, raw)
    print(f'{len(bundles)} studies, {len(records)} immutable objects', flush=True)


def finish():
    with tarfile.open(OUT / UPLOAD, 'r:gz') as archive:
        index = json.load(archive.extractfile('MANIFEST.json'))
    validation = json.loads(CHECK.read_bytes())
    if validation['upload_sha256'] != sha((OUT / UPLOAD).read_bytes()):
        raise ValueError('Validation was for a different upload')
    files = {}
    for name in (*BASE_FILES, *DOCS, UPLOAD):
        raw = (OUT / name).read_bytes()
        files[name] = dict(bytes=len(raw), sha256=sha(raw))
    manifest = dict(schema='ai2050_manual_release.v1', center_version='1.0.40', production_deployed=False,
                    independent_recomputation='pending', research_uploads=[UPLOAD], studies=index['studies'],
                    validation=validation, files=files)
    manifest_raw = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8') + b'\n'
    write(OUT / 'RELEASE_MANIFEST.json', manifest_raw)
    hashes = ''.join(f'{v["sha256"]}  {k}\n' for k, v in sorted(files.items()))
    hashes += f'{sha(manifest_raw)}  RELEASE_MANIFEST.json\n'
    write(OUT / 'SHA256SUMS.txt', hashes.encode('ascii'))
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted((*files, 'RELEASE_MANIFEST.json', 'SHA256SUMS.txt')):
            archive.write(OUT / name, f'{OUT.name}/{name}')
    with zipfile.ZipFile(data) as archive:
        for name, meta in files.items():
            if sha(archive.read(f'{OUT.name}/{name}')) != meta['sha256']:
                raise ValueError(f'Handoff byte identity mismatch: {name}')
        if archive.testzip():
            raise ValueError('Handoff ZIP integrity check failed')
    write(ARCHIVE, data.getvalue())
    write(ARCHIVE.with_suffix(ARCHIVE.suffix + '.sha256'), f'{sha(data.getvalue())}  {ARCHIVE.name}\n'.encode('ascii'))
    print(f'{ARCHIVE.name} {sha(data.getvalue())}', flush=True)


if __name__ == '__main__':
    {'data': build_data, 'finish': finish}[sys.argv[1]]()
