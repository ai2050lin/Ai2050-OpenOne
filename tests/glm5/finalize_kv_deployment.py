"""Write release checksums, clean this run's scratch, and append handoff records."""

from datetime import datetime
import io
import json
from pathlib import Path
import shutil
import zipfile

from b36_semantic14_memo import append
from package_kv_release import OUT, ROOT, UPLOADS, VERSION, WEB, WORK, sha, verify
from deploy.storage_policy import StorageBlocked, check_write, staging_file


PROOF=ROOT/'tests/glm5/result/deployment_20261117'
SUMMARY=ROOT/'research/glm5/docs/AGI_RESEARCH_SUMMARY_20261117.md'
GLM_MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
CLAUDE_MEMO=ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
WINDOWS=ROOT/'tests/glm5/result/goodfire_middle_20261113_v2/desktop/AI2050-Setup-1.0.15-x64.exe'
WINDOWS_SHA='5331d610de58543795ae4e485e090d072634e38e607fd00a3d78482a5dd590b2'
TITLE='中心 1.0.40 手动发布交付、当前机制证据与阶段计划'


def save(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    check_write(path,len(raw)).write_bytes(raw)


def cleanup():
    if not WORK.exists():return 0
    assert WORK.resolve()==WORK and WORK.parent==(ROOT/'tests/glm5_temp').resolve()
    owned=[WORK,*WORK.rglob('*')]
    assert all(not x.is_symlink() and not x.is_junction() and x.resolve().is_relative_to(WORK) for x in owned)
    size=sum(x.stat().st_size for x in owned if x.is_file())
    shutil.rmtree(WORK)
    return size


def combined_package(files):
    target=OUT.parent/f'AI2050-deployment-20261117-center-{VERSION}.zip'
    if target.exists():
        raise FileExistsError(target)
    with io.BytesIO() as image:
        with zipfile.ZipFile(image,'w',zipfile.ZIP_STORED) as z:
            for p in sorted(files):z.write(p,arcname=p.name)
        try:
            with image.getbuffer() as payload:
                check_write(target,len(payload)+384*2**10,operation='发布组合 ZIP')
                with staging_file(target) as partial:
                    with partial.open('wb') as stream:stream.write(payload)
                    with zipfile.ZipFile(partial) as z:
                        assert z.testzip() is None
                        assert {i.filename:sha(z.read(i)) for i in z.infolist()}=={p.name:sha(p.read_bytes()) for p in files}
        except StorageBlocked as error:
            print(error,flush=True)
            return dict(status='space',reason=str(error))
    digest=sha(target.read_bytes())
    checksum=(digest+'  '+target.name+'\n').encode()
    check_write(target.with_suffix('.zip.sha256'),len(checksum)).write_bytes(checksum)
    return dict(status='built',path=str(target),bytes=target.stat().st_size,sha256=digest)


def main():
    assert TITLE not in GLM_MEMO.read_text(encoding='utf-8-sig')
    verify()
    api=json.loads((PROOF/'local-api.json').read_text(encoding='utf-8'))
    browser=json.loads((PROOF/'browser.json').read_text(encoding='utf-8'))
    assert api['health']['version']==VERSION and api['web_http_files']==44
    assert len(browser['checks'])==3 and not browser['errors'] and not browser['modelRequests']
    assert '63 passed (63)' in (PROOF/'vitest.log').read_text(encoding='utf-8')
    assert 'pass 11' in (PROOF/'electron-tests-loopback.log').read_text(encoding='utf-8-sig')
    assert sha(WINDOWS.read_bytes())==WINDOWS_SHA
    summary=SUMMARY.read_bytes()
    assert not any(c<32 and c not in (9,10,13) for c in summary)
    check_write(OUT/SUMMARY.name,len(summary)).write_bytes(summary)
    generated={'RELEASE_MANIFEST.json','SHA256SUMS.txt'}
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
           for p in OUT.iterdir() if p.is_file() and p.name not in generated}
    manifest=dict(schema='ai2050_manual_release.v1',center_version=VERSION,web=WEB,
                  created_at=datetime.now().astimezone().isoformat(timespec='seconds'),research_uploads=UPLOADS,files=files,
                  production_deployed=False,evidence_promoted=False,
                  windows=dict(new_installer='storage_gate',existing_version='1.0.15',existing_sha256=WINDOWS_SHA),
                  verification=dict(renderer_tests=63,electron_tests=11,browser=browser,local_api=api))
    manifest_file=OUT/'RELEASE_MANIFEST.json'
    if manifest_file.exists():
        prior=json.loads(manifest_file.read_bytes())
        assert prior['files']==manifest['files'] and prior['center_version']==VERSION
        manifest=prior
    else:
        save(manifest_file,manifest)
    files['RELEASE_MANIFEST.json']={'sha256':sha((OUT/'RELEASE_MANIFEST.json').read_bytes())}
    sums=''.join(f'{v["sha256"]}  {k}\n' for k,v in sorted(files.items())).encode()
    check_write(OUT/'SHA256SUMS.txt',len(sums)).write_bytes(sums)
    assert all(sha((OUT/k).read_bytes())==v['sha256'] for k,v in files.items())
    freed=cleanup()
    cleanup_text=(f'{freed:,} 字节' if freed else '首次封包前已清理，构建和隔离中心测得文件合计至少 8,229,331 字节')
    combined=combined_package([*(OUT/k for k in files),OUT/'SHA256SUMS.txt'])
    zip_reference=Path(combined['path']).relative_to(ROOT).as_posix() if combined['status']=='built' else '发布目录逐件校验交付'
    body=f"""本阶段完成当前代码与研究数据的手动发布交接及研究证据总审。中心 VERSION 由 1.0.39 更新至 {VERSION}。发布目录在 §release/deployment-20261117/§，组合交付为 §{zip_reference}§。网页独立 ZIP 与中心 tar 中 44 个网页文件逐一同哈希，中心清单校验通过 340 个内容文件。含现有升级/回退脚本、对象登记脚本、Phase 2952 五对象、Phase 2955 四对象和 Phase 2954–2955 紧凑复核数据。2955 新增可浏览报告和 L25H18 观察卡，待独立复算、预测未过门与剂量失败保留。正式环境由用户手动更新，生产读回待执行。

验证包括前端 63/63、Electron 11/11、定向 ESLint、Vite 生产构建、独立中心 API 版本与 44 个静态文件读回、研究对象上传和重复登记、无授权阻断、缺失对象 404，以及桌面和移动浏览器。测试用 D 盘隔离中心，浏览器无模型运行调用。首轮普通沙盒限制本机回环测试，切换到只涉及本机的执行环境后通过。中心采用上限 16 MiB 的内存压缩暂存并按实际压缩长度检查磁盘。Windows 完整安装器入口要求 20 GiB 保留线外 4 GiB，本轮磁盘预检停止，已有 1.0.15 SHA-256 已重验。复核日志见 §tests/glm5/result/deployment_20261117/§。本轮独占暂存清理情况为 {cleanup_text}。保留以前证据、冻结数据和正式中心数据。

阶段科学总结在 §research/glm5/docs/AGI_RESEARCH_SUMMARY_20261117.md§。当前可靠拼图是固定参数对上下文做条件写入、实发 State 的 K/V 可被后续重用、访问及其下游效应依赖任务与竞争源。2955 的 828 个合格插入视图无 Receipt 质量超过 State，32 个新句式改码来自 4 个反角色家族，31 个干预对齐视图中 20 个被清 State Value 逆转，联合阻断后 17/17 可由独立恢复同一个原生 State Value 恢复插入码。新词反角色、预测推广及部分剂量对照未达冻结要求。RDC 沿用 Phase 2747 的有域部分转移和类型化三图谱公式。没有新增全局机制定律。

下一段先冻结非算术记录更新、自然生成与停止任务，用同内容变距离、同长度换内容和固定末位的控制明确因果范围。然后在新词、新句式、新组合分别比较词汇/位置/序列基线、原生坐标与组、等预算字典或归因，预测 State、来源分布及完整输出分布。最后在同一材料上检验逐例换词、匹配阻断、原生恢复、副作用以及独立物理节点复算，再串行扩展到 GLM-4-9B。大阶段靶标是可证伪且有未见预测能力的形成—保存—访问—输出机制。当前计划未执行新 GPU 测试。""".replace('§',chr(96))
    memo=append(check_write(GLM_MEMO),TITLE,body)
    link=append(check_write(CLAUDE_MEMO),'中心手动发布与当前研究计划索引',
                f'详见 GLM5 Phase {memo["phase"]}。中心 {VERSION} 和网页、Phase 2952/2955 增量及脚本交接见 release/deployment-20261117。系统总结及阶段设计见 research/glm5/docs/AGI_RESEARCH_SUMMARY_20261117.md。正式更新和生产读回待管理员执行。')
    save(PROOF/'delivery.json',dict(memos=[memo,link],manifest_sha256=files['RELEASE_MANIFEST.json']['sha256'],
                                  removed_scratch_bytes=freed,archive=combined,release=manifest))
    print('GLM5 Phase',memo['phase'],combined,flush=True)


if __name__=='__main__':
    main()
