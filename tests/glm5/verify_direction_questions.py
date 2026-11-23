"""Validate public direction designs and render the human-readable question book."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write

SHARED = ROOT / "shared/dist_protocol"
BOOK = ROOT / "research/glm5/docs/研究方向树与测试题.md"


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = value
    return result


def read(relative):
    return json.loads((SHARED / relative).read_bytes(), object_pairs_hook=unique)


def validate_pair(pair):
    if not pair["factor"].strip() or pair["prompt"].count("{x}") != 1:
        raise ValueError(f"Missing control factor or ambiguous slot: {pair}")
    variants = pair["variants"]
    if len(variants) != 2 or len({v["value"] for v in variants}) != 2:
        raise ValueError("Expected two distinct variants")
    if any(not v["value"].strip() or not v["expected"].strip() for v in variants):
        raise ValueError("A design input or reference answer is empty")


def render():
    atlas = read("lang_map/lang_map.json")
    catalog = read("test_records/direction_questions.json")
    pilot_ids = {r["id"] for r in read("test_records/pilot.json")["records"]}
    all_items = [item for d in atlas["domains"] for s in d["subs"] for item in s["items"]]
    for key, items in (("items", all_items), ("dimensions", atlas["dims"])):
        ids = [item["id"] for item in items]
        if len(set(ids)) != len(ids) or set(ids) != set(catalog[key]):
            raise ValueError(f"Direction inventory differs from language map: {key}")
        for item_id, plan in catalog[key].items():
            validate_pair(plan)
            if "followup" in plan:
                validate_pair(plan["followup"])
            for ref in plan.get("related", []):
                if ref["id"] not in pilot_ids or not ref["reason"].strip():
                    raise ValueError(f"Unresolvable reference in {item_id}")

    lines = ["# 研究方向树与测试题", "",
             f"设计题库 v{catalog['version']} · 语言图谱 v{atlas['version']}。每个稳定编号对应一个研究项或一项横切条件。",
             "本册公开展示的题目均为设计材料。A、B 各自列出预期答案。正式试验需要另建未曝光家族、冻结评分及通过条件。", "",
             "## 目录", ""]
    for domain in atlas["domains"]:
        lines.append(f"- **{domain['name']}**　" + "、".join(s["name"] for s in domain["subs"]))
    lines.extend(["- **横切条件**　" + "、".join(d["name"] for d in atlas["dims"]), "",
                  "## 怎样逐步填充", "",
                  "1. 先逐项查历史。为一项测试选择一个主编号，必要时引用其他研究项与横切条件。按来源记录历史关联，保持原题、结果、权限和暴露状态可追溯。同一次运行只计一次。",
                  "2. 找到原包后，另记运行 ID、题目与条件、模型快照、实际输出、评分口径、失败环节和原始证据。只有备忘录时保留出处，标为待核对。相关旧题的成绩不能填入本版新题。",
                  "3. 先补能直接区分两种解法的题族，例如知识属性与比较、多步关系与第二条边、主谓一致与相邻名词。例题 A/B 是起点。更改无关名字、顺序、答案标签，设置维持答案和翻转答案两类对照，按家族留出。语种、读出、模型和方法分别记录。",
                  "4. 新模型数据按项目流程先查重。行为在所有关键格可靠后，再为内部机制另列竞争预测、原生阻断、等强对照和恢复。设计覆盖、实际运行、材料完整度、科学结论分别计数。", "",
                  "A/B 是一个对照因素的两个取值。有些对照需要改写整段语境，可能同时改变词长、位置或词汇；进入机制研究前另行配平。未登记为锁箱的公开题都属已曝光。", ""])

    def section(item, plan):
        lines.extend([f"#### {item['name']} · `{item['id']}`", ""])
        for design in [plan] + ([plan["followup"]] if "followup" in plan else []):
            lines.extend([f"对照因素为{design['factor']}。", ""])
            for label, v in zip("AB", design["variants"]):
                lines.append(f"- **{label}**　{design['prompt'].replace('{x}', v['value'])}　参考答案：**{v['expected']}**")
            lines.append("")
            if "interpretation" in design:
                lines.extend([design["interpretation"], ""])
        for ref in plan.get("related", []):
            lines.extend([f"相关历史 `{ref['id']}`。{ref['reason']}", ""])
        if plan.get("centerFamily"):
            lines.extend([f"中心同族检索键为 `{plan['centerFamily']}`。登录后可查运行，需逐项核对是否适用于该题。", ""])

    for domain in atlas["domains"]:
        lines.extend([f"## {domain['name']} · {domain['id']}", ""])
        for sub in domain["subs"]:
            lines.extend([f"### {sub['name']}", ""])
            for item in sub["items"]:
                section(item, catalog["items"][item["id"]])
    lines.extend(["## 横切条件", ""])
    for dim in atlas["dims"]:
        section(dim, catalog["dimensions"][dim["id"]])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the derived Markdown book")
    args = parser.parse_args()
    content = render()
    if args.write:
        check_write(BOOK, len(content.encode("utf8")), operation="生成完整研究方向题册")
        BOOK.write_text(content, encoding="utf8")
    if BOOK.read_text(encoding="utf8") != content:
        raise SystemExit("Question book differs from the verified design. Regenerate with --write.")
    print("Verified direction coverage, A/B materials, reference identities and question book.")
