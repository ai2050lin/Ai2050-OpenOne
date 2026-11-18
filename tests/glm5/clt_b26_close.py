"""Publish the local reproducible research record and append both memo indexes."""
from datetime import datetime
import os
from pathlib import Path
import re

from clt_b26 import D, HERE, OUT, ROOT, SEED, save
from deploy.storage_policy import check_write
from tools.provenance import check_provenance, write_result

report = OUT / "report.zh-CN.md"
result = D.read(OUT / "analysis.json")
pre = D.read(OUT / "prereg.json")
assert not (OUT / "manifest.json").exists()
memo = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
index = ROOT / "research/claude/docs/AGI_CLAUDE_MEMO.md"


def next_phase(contents):
    return max(map(int,re.findall(rb"^## Phase (\d+):",contents,re.M))) + 1


def append(path, block, previous):
    raw = block.encode("utf-8")
    check_write(path,len(raw),operation="B26 append-only research record")
    assert path.read_bytes()==previous
    with path.open("ab") as stream:
        stream.write(raw)
    assert path.read_bytes()==previous+raw


before, prior_index = memo.read_bytes(), index.read_bytes()
phase, index_phase = next_phase(before), next_phase(prior_index)
when = datetime.now().strftime("%Y-%m-%d %H:%M")
heading = "Qwen14B B26H36 的行号、名字与正文取值干预"
append(memo,f"\n\n## Phase {phase}: {heading} [{when}]\n\n"+report.read_text(encoding="utf-8"),before)
append(index,f"\n\n## Phase {index_phase}: B26H36 行号与名字的原生干预 [{when}]\n\n"
       f"见 GLM5 Phase {phase}。新发现64家族、确认128家族，冻结后进行 h18、Q/K 原参数乘积、K 位置相位及正文 V 的匹配阻断与同输入恢复。\n",
       prior_index)

os.environ.update(GIT_CONFIG_COUNT="1",GIT_CONFIG_KEY_0="safe.directory",GIT_CONFIG_VALUE_0=str(ROOT))
receipt = write_result("clt_b26_native",dict(phase=phase,analysis=result),
                       model_path=pre["provenance"]["model_path"],seed=SEED,tool="clt_b26.py")
assert not check_provenance(D.read(receipt))
paths = {p for p in OUT.iterdir() if p.is_file()}
paths.update(HERE/n for n in pre["code"])
paths.update(HERE/n for n in ("clt_b26_report.py","clt_b26_plot.py","clt_b26_close.py","clt_b26_param_audit.py"))
paths.add(receipt)
files = {str(p.relative_to(ROOT)):D.sha(p) for p in sorted(paths)}
save("manifest.json",dict(provenance=pre["provenance"],phase=phase,index_phase=index_phase,
                          receipt=str(receipt.relative_to(ROOT)),files=files))
assert all(D.sha(ROOT/p)==digest for p,digest in files.items())
print("Appended phases",phase,index_phase,"and verified",len(files),"files")
