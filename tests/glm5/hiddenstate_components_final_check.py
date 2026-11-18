"""Append a typography correction without altering the scientific memo history."""
import ast,hashlib,re
from datetime import datetime
import hiddenstate_components as H


def main():
    plan=H.checked();record=H.read(H.OUT/'memo-append.json');delivery=H.read(H.OUT/'delivery-audit.json')
    memo=H.Path(record['memo']);old=memo.read_bytes();assert hashlib.sha256(old[:record['previous_bytes']]).hexdigest()==record['previous_sha256']
    marker=f"Phase {record['phase']} 公式排版勘误"
    if marker.encode() not in old:
        stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
        correction=rf'''

**{marker} [{stamp}]。** 收尾检查发现生成脚本把公式中两个 `\text` 的反斜杠当作制表转义；仅影响标签排版，不影响数值、冻结分析或结论。按append规则保留原文，正确公式如下。生成脚本已改用原始字符串，最终源码哈希另记final-check.json。

$$
\hat h_\ell(c,L)=\mu_\ell+c\,a_\ell+L\,b_\ell+cL\,d_\ell,
\qquad \text{{语言对照方向}}=b_\ell+c\,d_\ell,
\qquad \text{{条件中心}}=\mu_\ell+c\,a_\ell.
$$
'''
        assert memo.read_bytes()==old
        with memo.open('ab') as f:f.write(correction.encode('utf-8'))
        assert memo.read_bytes().startswith(old)
    for path in delivery['source_hashes']:
        if path.endswith('.py'):ast.parse((H.ROOT/path).read_text(encoding='utf-8'),filename=path)
    # Count the actual retained outputs and the two new CAS copies, excluding
    # mutable shared SQLite bookkeeping and pre-existing research data.
    reg=H.read(H.OUT/'registration.json')
    local_bytes=sum(f.stat().st_size for f in H.OUT.rglob('*') if f.is_file())
    import tarfile
    with tarfile.open(reg['bundle'],'r:gz') as tar:
        cas_bytes=2*sum(m.size for m in tar.getmembers() if m.name.startswith('objects/'))
    assert local_bytes+cas_bytes<40*2**20
    H.save('final-check.json',{'run_id':H.RUN,'phase':record['phase'],'index_phase':record['index_phase'],
        'memo_append_only':True,'typography_correction':True,'native_dependencies_still_frozen':True,
        'final_source_hashes':{path:H.sha(H.ROOT/path) for path in delivery['source_hashes']},
        'checker_sha256':H.sha(__file__),'new_results_and_cas_MiB':(local_bytes+cas_bytes)/2**20,
        'current_report_sha256':reg['report_sha256'],'no_formal_promotion':True},plan['model_snapshot'])
    print(f"Final checks passed; GLM5 Phase {record['phase']}, Claude Phase {record['index_phase']}; retained new outputs + CAS {(local_bytes+cas_bytes)/2**20:.2f} MiB")


if __name__=='__main__':main()
