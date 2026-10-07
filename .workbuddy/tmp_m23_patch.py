# -*- coding: utf-8 -*-
"""M2-3 patch: export-client 增投 registry/cases.json 与 registry/industry.json 到客户端数据目录。
保持 registry 单一事实源；导出物为只读投影。"""
import io

P = r"D:\AI2050\Ai2050-OpenOne\ai2050_research_os\scripts\researchctl.py"
REPORT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_m23_patch.txt"

with io.open(P, 'r', encoding='utf-8') as f:
    t = f.read()

# --- 锚 1：客户端数据目录常量 ---
A1_OLD = 'CLIENT_SNAPSHOT = WORKSPACE / "frontend" / "public" / "research_data" / "current" / "snapshot.json"'
A1_NEW = ('CLIENT_SNAPSHOT = WORKSPACE / "frontend" / "public" / "research_data" / "current" / "snapshot.json"\n'
          'CLIENT_CASES = WORKSPACE / "frontend" / "public" / "research_data" / "current" / "cases.json"\n'
          'CLIENT_INDUSTRY = WORKSPACE / "frontend" / "public" / "research_data" / "current" / "industry.json"')
n1 = t.count(A1_OLD)
assert n1 == 1, "anchor1 hit %d" % n1
t = t.replace(A1_OLD, A1_NEW)

# --- 锚 2：export-client 写入逻辑 ---
A2_OLD = '''    write_json(CLIENT_SNAPSHOT, snapshot)
    print(f"客户端 Snapshot 已导出: {CLIENT_SNAPSHOT.relative_to(WORKSPACE)}")
    return 0'''
A2_NEW = '''    write_json(CLIENT_SNAPSHOT, snapshot)
    print(f"客户端 Snapshot 已导出: {CLIENT_SNAPSHOT.relative_to(WORKSPACE)}")
    # M2: registry 只读投影（cases / industry），与 canonical snapshot 同目录导出
    cases_path = REGISTRY / "cases.json"
    industry_path = REGISTRY / "industry.json"
    if cases_path.is_file():
        write_json(CLIENT_CASES, load_json(cases_path))
        print(f"客户端 cases 投影已导出: {CLIENT_CASES.relative_to(WORKSPACE)}")
    else:
        print("警告: registry/cases.json 不存在，跳过 cases 投影", file=sys.stderr)
    if industry_path.is_file():
        write_json(CLIENT_INDUSTRY, load_json(industry_path))
        print(f"客户端 industry 投影已导出: {CLIENT_INDUSTRY.relative_to(WORKSPACE)}")
    else:
        print("警告: registry/industry.json 不存在，跳过 industry 投影", file=sys.stderr)
    return 0'''
n2 = t.count(A2_OLD)
assert n2 == 1, "anchor2 hit %d" % n2
t = t.replace(A2_OLD, A2_NEW)

with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(t)

# --- 回读复核 ---
t2 = io.open(P, encoding='utf-8').read()
ok1 = 'CLIENT_CASES = WORKSPACE' in t2
ok2 = '客户端 cases 投影已导出' in t2
ok3 = 'CLIENT_INDUSTRY' in t2
assert ok1 and ok2 and ok3, "readback failed: %s %s %s" % (ok1, ok2, ok3)
io.open(REPORT, 'w', encoding='utf-8').write("M2_3 anchors=2 readback=3/3 OK")
print("M2_3_PATCH_OK")
