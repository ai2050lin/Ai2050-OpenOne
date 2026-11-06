"""Append the large-archive delivery correction to the same research phase."""
import sys,json
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_publish import register_binding
from shared.dist_protocol.mechanism_trace import TraceStore,file_sha,digest
from deploy.research_library import Library

def main():
    r=P.read(P.OUT/'delivery-registration.json');lib=Library(P.ROOT/'distributed_data/research_library');report=lib.artifact(r['report_sha256']);snap=report['model_snapshot'];prov=report['provenance'];pack=P.read(P.OUT/'offline-stream-acceptance.json')
    logs={}
    for name,expect in [('software-tests-stream-final.log','Ran 14 tests'),('client-tests-stream-final.log','62 passed'),('protocol-tests-stream-final.log','Ran 261 tests'),('frontend-build-stream-final.log','built in')]:
        p=P.OUT/name;text=p.read_text(encoding='utf-8',errors='replace');assert expect in text and 'FAILED' not in text
        if name.startswith(('software','protocol')):assert '\nOK' in text
        logs[name]={'sha256':file_sha(p),'tail':'\n'.join(text.splitlines()[-10:])}
    memo=P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';marker='本阶段完整历史研究库的流式交付补记';old=memo.read_text(encoding='utf-8');stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    if marker not in old:
        note=f'''\n\n### {marker} [{stamp}]\n\n正式附加包生成后测得研究全版本库压缩544,559,482字节、解压1,796,228,136字节。旧客户端先整包转字符串再JSON.parse，无法持续承载这样的档案。新增gzip NDJSON逐条读取：中心每次仅解码一个产物，客户端经解压/UTF-8流逐条写入独立暂存前缀；schema、ID、版本、计数与完成标记全部通过后，才用IndexedDB事务一起切换元数据并清理旧版。截断、格式错误、重复路径和存储失败不发布新版本，暂存失败清理不会覆盖现有缓存。无可用IndexedDB时明确要求本地数据库，仍可在线浏览，不退化成把整库塞进内存。下载进度按实际字节显示，不拿旧压缩文件大小伪作流式传输分母。\n\n真实HTTP整库验收版本{pack['version']}：{pack['entries']}条接口记录、{pack['CAS_blobs_hash_verified']}个CAS产物逐一重算canonical SHA全部一致，包含当前报告、图谱、4358案例索引、验收与科研备忘。最大单条{pack['max_one_entry_characters']:,}字符，没有构造整包字符串；传输{pack['compressed_bytes_streamed']:,}字节、耗时{pack['seconds']:.2f}秒。流式重编码压缩率与原存档不同。原519MiB压缩JSON仍保留供旧工具读取，正式新客户端研究库下载使用流式格式。首次64KiB解析版本吞吐较低，改为1MiB分块后完成上述真实整库校验；未把中止的慢请求记为成功。\n\n最终14项研究/流式软件控件、62项客户端、261项协议和11项轨迹回归共348项通过，前端构建成功；新增测试覆盖跨块Unicode、大条目、截断、错误完成计数/版本、存储回调失败。实测整库HTTP流已验证；未声称在浏览器另完成1.8GB的全量IndexedDB落盘和跨设备同步验证。科研主判与模型结果不变，本补记只完善持续积累成果的交付能力；后续仍需事实×问题×答案编码和图谱留出预测的大阶段。\n'''
        with memo.open('a',encoding='utf-8',newline='\n') as f:f.write(note)
    def archive(name,doc,run=P.RUN):
        d={'schema':'research_native_evidence.v1','kind':'real','run_id':run,'model_snapshot':snap,'provenance':prov,**doc};p=P.PUBLIC/(name+'-'+digest(d)[:12]+'.research.json');P.save(p,d)
        return lib.ingest(d,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    files=[P.ROOT/n for n in ['deploy/research_pack_stream.py','deploy/platform_service.py','frontend/src/components/app/rdc_fusion/dataStore.js','frontend/src/components/app/rdc_fusion/researchStream.js','frontend/src/components/open/DataBar.jsx','tests/glm5/test_native_mlp_function.py','tests/glm5/native_mlp_function_stream.test.jsx','tests/glm5/native_mlp_function_pack_acceptance.py','tests/glm5/native_mlp_function_stream_delivery.py']]
    source=archive('stream-delivery-source',{'schema':'research_source_bundle.v1','files':[{'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p),'text':p.read_text(encoding='utf-8')} for p in files]})
    mr=P.read(P.OUT/'memo-registration.json');text=memo.read_text(encoding='utf-8');begin=text.index('## Phase '+str(mr['phase'])+':');ms=archive('scientific-memo-complete',{'schema':'research_scientific_memo.v1','phase':mr['phase'],'text':text[begin:],'path':str(memo.relative_to(P.ROOT)),'file_sha256_at_capture':file_sha(memo)})
    qa=archive('stream-delivery-acceptance',{'schema':'research_client_acceptance.v1','previous_acceptance_sha256':r['acceptance_sha256'],'tests':348,'checks':logs,'real_full_stream':pack,'source_bundle_sha256':source,'scientific_memo_sha256':ms,'browser_full_1_8GB_indexeddb_ingestion_tested':False,'native_tensor_packaged':False})
    with lib.db() as db:rows=db.execute("SELECT sha FROM artifacts WHERE kind='space_binding' ORDER BY created DESC,sha DESC").fetchall()
    b=next(lib.artifact(row['sha']) for row in rows if lib.artifact(row['sha'])['run_id']==P.RUN)
    register_binding(archive,TraceStore(),P.RUN,r['report_sha256'],r['graph_sha256'],b['traces'],{**b['context'],'client_acceptance_sha256':qa,'scientific_memo_sha256':ms},'原生MLP：答案方向反例与真实3D闭环',native=r['native'])
    wp=next(w['body'] for w in lib.workpackages()['workpackages'] if w['body']['run_id']==P.RUN)
    archive('workpackage-stream-delivered',{**wp,'stages':{**wp['stages'],'client_acceptance':{'status':'done','artifact_sha256':qa,'tests':348,'browser_visual_verified':True,'full_offline_stream_verified':True},'scientific_memo':{'status':'appended','phase':mr['phase'],'artifact_sha256':ms}}})
    P.save(P.OUT/'delivery-registration-stream.json',{**r,'acceptance_sha256':qa,'scientific_memo_sha256':ms,'stream_source_bundle_sha256':source,'full_stream_verified_version':pack['version']})
    print(json.dumps({'tests':348,'acceptance_sha256':qa,'memo_sha256':ms,'full_stream_verified':True}))
if __name__=='__main__':main()
