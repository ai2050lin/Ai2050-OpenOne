# -*- coding: utf-8 -*-
"""atlas_cli.py — TMA Atlas 账本 / 特征注册表的公共验证 CLI（零 GPU，只读）。

目的（审计 design/platform_gap_review_v1.md P0-3）：让任何人一条命令复核账本与
特征锚点，把"失败入账 + 哈希链"变成可公共验证的资产。

子命令
  chain               全量验证 atlas_ledger.json 哈希链 + 全部 .bak 快照（append-only 证明）
  registry            验证 atlas_registry_v1_*.json 版本链（supersedes 连通性 + 文件 sha8）
  show    FTR-XX      打印特征卡（statement / values / anchors / counter_evidence）
  verify  FTR-XX      逐锚验证：定位封存产物 → 自声明 sha8 绑定 → 内容哈希重算（已知配方）
                      → 数值断言重算比对（float 容差 1e-9 相对）

哈希配方依据（考古结论，2026-10-27）
  - ledger 链头（权威，p3280_disk_verify v8 实证）：
      head = sha256( json.dumps(doc, ensure_ascii=False, indent=1) )[:8]
      其中 doc 为全量账本、且 ledger_sha256_8 字段置为【上一链头】；
      每条 measurement 的 prev_post == 加入它之前的链头。
  - result 内容哈希（p3280 v1 实证）：sha256(dumps(result 去 res_sha8, sort_keys=True,
      ensure_ascii=False))[:8]。早期 phase（如 3154）的 res 配方已不可复现
      （生产脚本内嵌、中途构造），CLI 如实报告 HASH_RECIPE_UNKNOWN，不伪判 FAIL；
      此时绑定依赖「文件自声明 res_sha8/seal_sha8 == 锚断言」+ 数值重算。

用法（仓库根目录）
  python scripts/atlas_cli.py chain
  python scripts/atlas_cli.py registry
  python scripts/atlas_cli.py verify FTR-01
  python scripts/atlas_cli.py show   FTR-16
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json")
RESULT_BASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")

FAILS: list[str] = []
WARNS: list[str] = []
CHECKS = [0]


def check(name: str, cond: bool, detail: str = "") -> bool:
    CHECKS[0] += 1
    ok = bool(cond)
    if not ok:
        FAILS.append(name)
    print(("PASS " if ok else "FAIL ") + name + ((" | " + detail) if detail else ""), flush=True)
    return ok


def sha8_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


def _head(doc_with_self_as_prev: dict) -> str:
    return sha8_bytes(json.dumps(doc_with_self_as_prev, ensure_ascii=False, indent=1).encode("utf-8"))


# ---------------------------------------------------------------- chain ----

def cmd_chain(verify_baks: bool = True) -> int:
    doc = json.load(open(LEDGER, encoding="utf-8"))
    ms = doc["measurements"]
    stored_head = doc.get("ledger_sha256_8")
    n = len(ms)
    print(f"ledger: n_measurements={n} stored_head={stored_head}")

    # 1) 链头复算（p3280 v8 口径：全量账本 + 自字段=上一链头）
    probe = json.loads(json.dumps(doc))
    probe["ledger_sha256_8"] = ms[-1].get("prev_post")
    check("chain.head_recompute", _head(probe) == stored_head,
          f"recomputed={_head(probe)} stored={stored_head}")

    # 2) prev_post 覆盖：链纪应为「尾部连续段」；段内缺口 = 真实记账缺陷（如实 FAIL）
    prev_idx = [i for i, m in enumerate(ms) if "prev_post" in m]
    era_start = prev_idx[0] if prev_idx else n
    in_era_missing = [i for i in range(era_start, n) if "prev_post" not in ms[i]]
    if prev_idx:
        for i in in_era_missing:
            print(f"  链纪缺口: measurement[{i}] phase={ms[i].get('phase')} "
                  f"name={ms[i].get('name')} 无 prev_post（该条目加入时未写链字段）")
    check("chain.prev_post_tail_contiguous", prev_idx and not in_era_missing,
          f"prev_post 覆盖 {len(prev_idx)}/{n} 条（index {era_start}..{n - 1}），段内缺口={in_era_missing}")

    # 3) .bak 快照：分纪元验证
    #    - 链纪快照（末条含 prev_post）：前缀性 + 自哈希必须过；
    #    - legacy 纪元快照（N 线回填 / v1 期，见 migration_history）：就地迁移致前缀/头
    #      不可比，仅报告不计 FAIL（migration_history 明载当年以 per-file sha8 代替链头）。
    baks = []
    if verify_baks:
        baks = sorted(glob.glob(os.path.join(ROOT, "research", "gpt5", "atlas", "atlas_ledger.json.bak*")))
        era_bad, era_total, unparse, era_ok = [], 0, 0, 0
        legacy_diff, legacy_ok, legacy_total = [], 0, 0
        bak_heads = {}
        for bp in baks:
            try:
                b = json.load(open(bp, encoding="utf-8"))
            except (OSError, ValueError):
                unparse += 1
                continue
            bms = b.get("measurements", [])
            bak_heads[len(bms)] = (os.path.basename(bp), b)
            if bms and "prev_post" in bms[-1]:
                era_total += 1
                if bms != ms[: len(bms)]:
                    era_bad.append(os.path.basename(bp) + " (prefix)")
                    continue
                probe = json.loads(json.dumps(b))
                probe["ledger_sha256_8"] = bms[-1]["prev_post"]
                if _head(probe) != b.get("ledger_sha256_8"):
                    era_bad.append(os.path.basename(bp) + " (head)")
                else:
                    era_ok += 1
            else:
                legacy_total += 1
                if bms == ms[: len(bms)]:
                    legacy_ok += 1
                else:
                    legacy_diff.append(f"{os.path.basename(bp)}(n={len(bms)})")
        check("bak.chain_era_prefix_and_head", era_total == 0 or (not era_bad and era_ok == era_total),
              f"链纪快照 {era_ok}/{era_total} 通过，问题={era_bad}")
        print(f"  快照总数={len(baks)}（unparseable={unparse}）：链纪 {era_ok}/{era_total} 通过；"
              f"legacy {legacy_ok}/{legacy_total} 为前缀，就地迁移差异={legacy_diff}")
        # 跨档衔接：链纪快照的 head 应 == 主账本对应下一条的 prev_post
        link_checked = False
        for cnt in sorted(bak_heads, reverse=True):
            if cnt >= era_start and cnt < n and "prev_post" in ms[cnt]:
                name, b = bak_heads[cnt]
                check("chain.bak_head_links_main", b["ledger_sha256_8"] == ms[cnt]["prev_post"],
                      f"{name}(n={cnt}) head={b['ledger_sha256_8']} == ms[{cnt}].prev_post")
                link_checked = True
                break
        if not link_checked:
            print("NOTE chain.bak_head_links_main: 无可衔接的链纪快照（信息性）")
    print(f"\n== chain: {CHECKS[0] - len(FAILS)}/{CHECKS[0]} PASS ==")
    return 0 if not FAILS else 1


# ------------------------------------------------------------- registry ----

def _registry_files() -> list[str]:
    return sorted(glob.glob(os.path.join(RESULT_BASE, "**", "atlas_registry_v1_*.json"), recursive=True))


def cmd_registry() -> int:
    files = _registry_files()
    print(f"registry files: {len(files)}")
    recs = []
    for fp in files:
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except (OSError, ValueError) as e:
            check(f"registry.parse {os.path.basename(fp)}", False, str(e))
            continue
        recs.append({"path": fp, "doc": d, "sha8": sha8_bytes(open(fp, "rb").read()),
                     "version": d.get("version"), "supersedes": d.get("supersedes")})
    for r in recs:
        print(f"  v{r['version']:<6} {r['sha8']}  {os.path.relpath(r['path'], RESULT_BASE)}")
    by_ver = {r["version"]: r for r in recs if r["version"] is not None}
    check("registry.version_unique", len(by_ver) == len(recs), f"{len(recs)} 个文件")
    vers = sorted(by_ver, key=lambda v: float(str(v).split(".")[-1]))
    minors = [int(str(v).split(".")[-1]) for v in vers]
    check("registry.versions_contiguous", minors == list(range(minors[0], minors[-1] + 1)),
          f"{vers[0]}..{vers[-1]}")
    bad = []
    for v, mnr in zip(vers, minors):
        if mnr == minors[0]:
            continue
        sup = str(by_ver[v]["doc"].get("supersedes") or "")
        prev_name = f"atlas_registry_v1_{mnr - 1}.json"
        if prev_name not in sup:
            bad.append((v, sup[:60]))
    check("registry.supersedes_chain", not bad, f"断链={bad}")
    mono = True
    for a, b in zip(vers, vers[1:]):
        fa = {f["id"] for f in by_ver[a]["doc"]["features"]}
        fb = {f["id"] for f in by_ver[b]["doc"]["features"]}
        if not fa <= fb:
            mono = False
            print(f"  feature id 回退: v{a} 有而 v{b} 无 -> {sorted(fa - fb)[:6]}")
    check("registry.feature_ids_monotonic", mono)
    last = by_ver[vers[-1]]["doc"]
    print(f"  latest v{vers[-1]}: features={len(last['features'])} failures={len(last['failures'])} "
          f"design_sha8={last.get('design_sha8')}")
    print(f"\n== registry: {CHECKS[0] - len(FAILS)}/{CHECKS[0]} PASS ==")
    return 0 if not FAILS else 1


# ---------------------------------------------------------------- show ----

def _latest_registry() -> dict:
    files = _registry_files()
    best = None
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        v = d.get("version") or -1
        if best is None or v > best[0]:
            best = (v, d)
    if best is None:
        raise SystemExit("找不到 atlas_registry_v1_*.json")
    return best[1]


def _feature(doc: dict, fid: str) -> dict:
    for f in doc["features"]:
        if f["id"].upper() == fid.upper():
            return f
    raise SystemExit(f"registry 无 {fid}；可用: {', '.join(f['id'] for f in doc['features'])}")


def cmd_show(fid: str) -> int:
    f = _feature(_latest_registry(), fid)
    print(json.dumps(f, ensure_ascii=False, indent=1))
    return 0


# --------------------------------------------------------------- verify ----

def _flatten(obj, prefix=""):
    """dict/list -> [(path, key, value)]；列表额外产出点号索引变体（ctx_effect.k1.7 ↔ [7]）。"""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.append((p, str(k), v))
            out.extend(_flatten(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.append((p, str(i), v))
            out.append((prefix + "." + str(i), str(i), v))
            out.extend(_flatten(v, p))
    return out


def _resolve_assert(art, akey: str):
    """锚键 → 产物值列表。支持两种口径：
    1) 语义路径 node:{ID}.{status|n_checks|n_pass}（audit 类，nodes[] 数组）；
    2) 扁平路径匹配（整路径相等或以 .akey 结尾）。"""
    if akey.startswith("node:"):
        rest = akey[len("node:"):]
        nid, _, field = rest.partition(".")
        vals = []
        for nd in (art.get("nodes") or []) if isinstance(art, dict) else []:
            if isinstance(nd, dict) and nd.get("id") == nid:
                if field == "status":
                    vals.append(nd.get("node_status", nd.get("status")))
                elif field == "n_checks":
                    vals.append(len(nd.get("checks") or []))
                elif field == "n_pass":
                    vals.append(sum(1 for c in (nd.get("checks") or []) if c and c[0]))
                else:
                    vals.append(nd.get(field))
        uniq = {json.dumps(v, sort_keys=True): v for v in vals}
        return list(uniq.values()), vals[:1]
    if akey.startswith("gate_"):
        # 锚键 gate_X ↔ 产物 gates.X（gates 字典结构）
        uniq2, paths2 = _match_assert_key(_flatten(art), "gates." + akey[len("gate_"):])
        if uniq2:
            return uniq2, paths2
    uniq, paths = _match_assert_key(_flatten(art), akey)
    return uniq, paths


def _match_assert_key(leaves, akey: str):
    """锚键匹配：整路径相等、以 .akey 结尾，或末段逆序（如锚 kappa.{model} ↔ 产物 {model}.kappa）。"""
    segs = akey.split(".")
    rev = ".".join(reversed(segs))
    vals, paths = {}, []
    for p, _, v in leaves:
        if p == akey or p.endswith("." + akey) or p.endswith("." + rev):
            vals[json.dumps(v, sort_keys=True)] = v
            paths.append(p)
    uniq = list(vals.values())
    return uniq, paths[:3]


def _content_hash_candidates(path: str) -> dict:
    """对 json 产物尝试已知内容哈希配方，返回 {recipe: sha8}。"""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {"file_bytes": sha8_bytes(open(path, "rb").read())}
    if isinstance(d, dict):
        variants = {
            "content_minus_res_sha8": {k: v for k, v in d.items() if k != "res_sha8"},
            "content_minus_res_seal": {k: v for k, v in d.items() if k not in ("res_sha8", "seal_sha8")},
            "content_minus_res_created": {k: v for k, v in d.items()
                                          if k not in ("res_sha8", "created")},
        }
        for name, dd in variants.items():
            out[name] = sha8_bytes(json.dumps(dd, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return out


Q06_DIR = os.path.join(ROOT, "tests", "deepseek_temp", "Phase40")
# 跨线 src 别名（产物按归属纪律留在原线；见 R7/R8 备忘录登记）
SRC_ALIASES = {
    "q03": [os.path.join(ROOT, "tests", "deepseek", "result", "**", "q03_result.json")],
    "q05": [os.path.join(ROOT, "tests", "deepseek", "result", "**", "q05_result.json")],
    "agate": [os.path.join(ROOT, "research", "deepseek", "atlas", "a_gate_closure_v1.json")],
}


def _verify_q06(asserts: dict):
    """q06（C_steer 基座测量，Phase 40 / deepseek 线归档）专用重算器。
    从封存 q06_cells_detail.json（441 cell × steer/rand × 5 剂量）确定性重算：
      - C_steer_main.value / .rand_value = best (sign,dose) 池化成功率
        （eligible 且 argmax==true_class 且 collat==0；口径 tests/deepseek/Phase40/q06_steer_base.py）；
      - .wilson.1 = Wilson 95% 上界（z=1.96 精确值，生产脚本同款）；
      - collateral.frac_zero = 全部 (cell, arm) 对中 collat==0 的份额（含 steer 与 rand）；
      - res_sha8 绑定：q06_report.txt 自声明行。
    返回 {akey: (ok, detail)}；不支持的键缺省。"""
    out = {}
    cells = json.load(open(os.path.join(Q06_DIR, "q06_cells_detail.json"), encoding="utf-8"))
    report = open(os.path.join(Q06_DIR, "q06_report.txt"), encoding="utf-8").read()

    def pool(prefix: str):
        eligible = [c for c in cells.values() if c.get("eligible")]
        best_rate, best_n = None, len(eligible)
        for sign in ("+1", "-1"):
            for dose in ("0.05", "0.10", "0.15", "0.25", "0.50"):
                key = f"{prefix}|{sign}|{dose}"
                succ = sum(1 for c in eligible
                           if isinstance(c.get(key), dict)
                           and c[key].get("argmax") == c.get("true_class")
                           and c[key].get("collat") == 0)
                rate = succ / len(eligible) if eligible else None
                if best_rate is None or (rate is not None and rate > best_rate):
                    best_rate = rate
        return best_rate, len(eligible)

    main_rate, n_el = pool("steer")
    rand_rate, _ = pool("rand")
    z = 1.96
    p = main_rate or 0.0
    den = 1 + z * z / n_el
    ctr = (p + z * z / (2 * n_el)) / den
    hw = z * math.sqrt(p * (1 - p) / n_el + z * z / (4 * n_el * n_el)) / den
    wilson_hi = ctr + hw
    allc = [v["collat"] for c in cells.values() for v in c.values()
            if isinstance(v, dict) and "collat" in v]
    frac0 = sum(1 for c in allc if c == 0) / len(allc)
    rec = {"C_steer_main.value": main_rate, "C_steer_main.rand_value": rand_rate,
           "C_steer_main.wilson.1": wilson_hi, "collateral.frac_zero": frac0}
    for akey, want in asserts.items():
        if akey == "res_sha8":
            ok = f"res_sha8={want}" in report
            out[akey] = (ok, f"q06_report.txt 自声明 res_sha8={want} 命中={ok}")
        elif akey in rec and rec[akey] is not None:
            ok = abs(float(rec[akey]) - float(want)) <= 1e-12
            out[akey] = (ok, f"重算={rec[akey]!r} want={want!r} |d|={abs(rec[akey] - want):.3e}")
        else:
            out[akey] = (None, "q06 重算器不支持该键")
    return out


def _registry_only_value(akey: str, want) -> list[str]:
    """键在封存 result 产物缺失时，查它是否曾被写进注册表快照（REGISTRY_ONLY 证据降级）。"""
    hits = []
    for fp in _registry_files():
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        uniq, _ = _resolve_assert(d, akey)
        for v in uniq:
            try:
                if (abs(float(v) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
                        if isinstance(want, (int, float)) and not isinstance(want, bool) else v == want):
                    hits.append("v" + str(d.get("version")))
                    break
            except (TypeError, ValueError):
                continue
    return hits


def cmd_verify(fid: str) -> int:
    f = _feature(_latest_registry(), fid)
    print(f"verify {f['id']} [{f.get('evidence_level')}] {f.get('statement', '')[:80]}...")
    all_ok = True
    for ai, anc in enumerate(f.get("anchors", [])):
        asserts = anc.get("asserts") or {}
        phase = anc.get("phase")
        pdir = os.path.join(RESULT_BASE, f"phase{phase}")
        print(f"\n-- anchor[{ai}] phase={phase} role={anc.get('role')} src={anc.get('src')} "
              f"tags={anc.get('tags')} asserts={len(asserts)}")
        if anc.get("src") == "q06":
            # 跨线归档（deepseek 线 Phase 40）：专用重算器，从封存 cells 逐位重算
            check(f"a{ai}.q06_dir", os.path.isdir(Q06_DIR), os.path.relpath(Q06_DIR, ROOT))
            res = _verify_q06(asserts)
            anchor_ok = True
            for akey, (ok, detail) in res.items():
                if ok is None:
                    print(f"NOTE a{ai}.{akey}: {detail}")
                    continue
                check(f"a{ai}.{akey}", ok, detail)
                anchor_ok = anchor_ok and bool(ok)
            all_ok = all_ok and anchor_ok
            continue
        alias = SRC_ALIASES.get(str(anc.get("src") or ""))
        cands = []
        if alias:
            for pat in alias:
                for fp in glob.glob(pat, recursive=True):
                    try:
                        d = json.load(open(fp, encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if isinstance(d, dict):
                        cands.append((fp, d))
            if not cands:
                check(f"a{ai}.alias_artifacts", False, f"别名产物缺失: {pat}")
                all_ok = False
                continue
        elif not os.path.isdir(pdir):
            check(f"a{ai}.phase_dir", False, f"缺失 {os.path.relpath(pdir, ROOT)}")
            all_ok = False
            continue
        else:
            for root, _, fs in os.walk(pdir):
                for fn in fs:
                    if not fn.endswith(".json"):
                        continue
                    fp = os.path.join(root, fn)
                    try:
                        d = json.load(open(fp, encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if isinstance(d, dict):
                        cands.append((fp, d))
        # 绑定：文件自声明 res_sha8/seal_sha8 命中锚断言
        want_res = asserts.get("res_sha8") or asserts.get("result_sha8")
        want_seal = asserts.get("seal_sha8")
        bound = None
        for fp, d in cands:
            if want_res and d.get("res_sha8") == want_res:
                bound = (fp, d)
                break
        if bound is None and want_seal:
            for fp, d in cands:
                if d.get("seal_sha8") == want_seal:
                    bound = (fp, d)
                    break
        if bound is None:
            # 无 sha8 绑定（audit 类锚）：按「断言命中率」绑定到最佳候选
            best, best_score = None, -1
            for fp, d in cands:
                score = 0
                for akey, want in asserts.items():
                    if not isinstance(want, (int, float, bool, str)):
                        continue
                    uniq, _ = _resolve_assert(d, akey)
                    if uniq:
                        got = uniq[0]
                        try:
                            ok = (len(uniq) == 1 and
                                  (abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
                                   if isinstance(want, (int, float)) and not isinstance(want, bool)
                                   else got == want))
                        except (TypeError, ValueError):
                            ok = False
                        score += 1 if ok else 0
                if score > best_score:
                    best, best_score = (fp, d), score
            n_val = sum(1 for w in asserts.values() if isinstance(w, (int, float, bool, str)))
            if best and best_score == n_val and n_val > 0:
                bound = best
        if bound is None:
            check(f"a{ai}.artifact_bound", False,
                  f"phase{phase} 下无自声明 res_sha8={want_res} 的 json（{len(cands)} 个候选）")
            all_ok = False
            continue
        fp, art = bound
        check(f"a{ai}.artifact_bound", True, os.path.relpath(fp, RESULT_BASE))
        # 内容哈希重算（已知配方；未命中报 RECIPE_UNKNOWN 不判 FAIL）
        hashes = _content_hash_candidates(fp)
        recipe_hit = None
        if want_res:
            for name, h in hashes.items():
                if name != "file_bytes" and h == want_res:
                    recipe_hit = name
                    break
            if recipe_hit:
                check(f"a{ai}.content_hash[{recipe_hit}]", True, f"== {want_res}")
            else:
                print(f"NOTE a{ai}.content_hash: 已知配方均未复现 {want_res} "
                      f"（早期 phase 配方内嵌于生产脚本）→ 记 HASH_RECIPE_UNKNOWN，绑定依赖文件自声明 "
                      f"res_sha8 + 磁盘 sha8={hashes.get('file_bytes')}")
        # 数值断言
        for akey, want in asserts.items():
            if akey in ("res_sha8", "seal_sha8", "result_sha8"):
                continue
            if not isinstance(want, (int, float, bool, str)):
                continue
            uniq, paths = _resolve_assert(art, akey)
            if not uniq:
                ro = _registry_only_value(akey, want)
                if ro:
                    msg = (f"REGISTRY_ONLY：产物无此键，但值存在于注册表快照 {'/'.join(sorted(set(ro)))}"
                           f"（当时入账未回写封存 result）→ 证据降级记录")
                    print(f"NOTE a{ai}.{akey}: {msg}")
                    WARNS.append(f"{fid} a{ai}.{akey}")
                    continue
                check(f"a{ai}.{akey}", False, "产物中未找到该键 (MISSING)")
                all_ok = False
            elif len(uniq) > 1:
                check(f"a{ai}.{akey}", False, f"多值歧义 {uniq[:3]} @ {paths}")
                all_ok = False
            else:
                got = uniq[0]
                if isinstance(want, float) or isinstance(got, float):
                    ok = abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
                    detail = f"got={got!r} want={want!r} |d|={abs(float(got) - float(want)):.3e}"
                else:
                    ok = got == want
                    detail = f"got={got!r} want={want!r}"
                check(f"a{ai}.{akey}", ok, detail)
                all_ok = all_ok and ok
        # registry.values 与 primary 锚一致性
        if anc.get("role") == "primary_measurement" and isinstance(f.get("values"), dict):
            for vk, vv in f["values"].items():
                uniq, paths = _resolve_assert(art, vk)
                if uniq:
                    ok = len(uniq) == 1 and abs(float(uniq[0]) - float(vv)) <= 1e-9 * max(1.0, abs(float(vv)))
                    check(f"a{ai}.values.{vk}", ok, f"registry={vv!r} artifact={uniq[:1]}")
                    all_ok = all_ok and ok
    print(f"\n== verify {fid}: {CHECKS[0] - len(FAILS)}/{CHECKS[0]} PASS, {len(WARNS)} NOTE ==")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="TMA Atlas 账本/注册表公共验证 CLI（只读）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chain", help="验证 ledger 哈希链 + .bak 快照")
    sub.add_parser("registry", help="验证 registry 版本链")
    p_show = sub.add_parser("show", help="打印特征卡")
    p_show.add_argument("fid")
    p_ver = sub.add_parser("verify", help="逐锚验证特征")
    p_ver.add_argument("fid")
    a = ap.parse_args()
    if a.cmd == "chain":
        return cmd_chain()
    if a.cmd == "registry":
        return cmd_registry()
    if a.cmd == "show":
        return cmd_show(a.fid)
    if a.cmd == "verify":
        return cmd_verify(a.fid)
    return 2


if __name__ == "__main__":
    sys.exit(main())
