#!/usr/bin/env python3
"""Explore SemEval-2007 lexical-substitution material without writing artifacts."""

from __future__ import annotations

import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for


XML_URL = "https://ltdata1.informatik.uni-hamburg.de/lexsub2016/Tasks/SemEval2007/test/lexsub_test.xml"
GOLD_URL = "https://ltdata1.informatik.uni-hamburg.de/lexsub2016/Tasks/SemEval2007/test/gold.gold"
MODELS = ("qwen3", "glm4", "deepseek7b")
WORD_RE = re.compile(r"^[a-z]+$")


def parse_gold(raw: bytes) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split(" :: ", 1)
        instance_id = int(left.rsplit(" ", 1)[1])
        counts: dict[str, int] = {}
        for item in right.split(";"):
            item = item.strip()
            if not item:
                continue
            value, count = item.rsplit(" ", 1)
            counts[value.lower()] = int(count)
        result[instance_id] = counts
    return result


def parse_xml(raw: bytes) -> dict[str, list[dict]]:
    root = ET.fromstring(raw)
    result: dict[str, list[dict]] = defaultdict(list)
    for lexelt in root.findall("lexelt"):
        item = lexelt.attrib["item"]
        lemma, pos = item.rsplit(".", 1)
        for instance in lexelt.findall("instance"):
            context = instance.find("context")
            if context is None:
                continue
            head = context.find("head")
            if head is None or head.text is None:
                continue
            prefix = context.text or ""
            suffix = head.tail or ""
            result[item].append({
                "instance_id": int(instance.attrib["id"]),
                "lemma": lemma.lower(),
                "pos": pos,
                "head": head.text,
                "prefix": prefix,
                "suffix": suffix,
                "sentence": f"{prefix}{head.text}{suffix}",
            })
    return result


def main() -> None:
    xml_raw = urllib.request.urlopen(XML_URL, timeout=60).read()
    gold_raw = urllib.request.urlopen(GOLD_URL, timeout=60).read()
    gold = parse_gold(gold_raw)
    instances = parse_xml(xml_raw)
    tokenizers = {name: tokenizer_for(name) for name in MODELS}

    def common_single_token(word: str) -> bool:
        if not WORD_RE.fullmatch(word):
            return False
        return all(len(tok.encode(" " + word, add_special_tokens=False)) == 1 for tok in tokenizers.values())

    panels = []
    for item, rows in sorted(instances.items()):
        lemma, pos = item.rsplit(".", 1)
        clean = []
        for row in rows:
            if row["head"].lower() != lemma.lower():
                continue
            words = re.findall(r"[A-Za-z]+", row["sentence"])
            if not 8 <= len(words) <= 100:
                continue
            if len(re.findall(r"[A-Za-z]+", row["prefix"])) < 8:
                continue
            clean.append(row)
        candidates = sorted({
            value
            for row in clean
            for value, count in gold.get(row["instance_id"], {}).items()
            if count >= 2 and value != lemma and common_single_token(value)
        })
        best = None
        for i, candidate_a in enumerate(candidates):
            for candidate_b in candidates[i + 1:]:
                a_rows = [
                    row for row in clean
                    if gold.get(row["instance_id"], {}).get(candidate_a, 0) >= 2
                    and gold.get(row["instance_id"], {}).get(candidate_b, 0) == 0
                    and not re.search(rf"\b{re.escape(candidate_a)}\b|\b{re.escape(candidate_b)}\b", row["prefix"] + row["suffix"], re.I)
                ]
                b_rows = [
                    row for row in clean
                    if gold.get(row["instance_id"], {}).get(candidate_b, 0) >= 2
                    and gold.get(row["instance_id"], {}).get(candidate_a, 0) == 0
                    and not re.search(rf"\b{re.escape(candidate_a)}\b|\b{re.escape(candidate_b)}\b", row["prefix"] + row["suffix"], re.I)
                ]
                if len(a_rows) < 2 or len(b_rows) < 2:
                    continue
                a_rows.sort(key=lambda row: (-gold[row["instance_id"]][candidate_a], row["instance_id"]))
                b_rows.sort(key=lambda row: (-gold[row["instance_id"]][candidate_b], row["instance_id"]))
                strength = sum(gold[row["instance_id"]][candidate_a] for row in a_rows[:2])
                strength += sum(gold[row["instance_id"]][candidate_b] for row in b_rows[:2])
                key = (strength, min(len(a_rows), len(b_rows)), candidate_a, candidate_b)
                if best is None or key > best[0]:
                    best = (key, candidate_a, candidate_b, a_rows[:2], b_rows[:2])
        if best is not None:
            _, candidate_a, candidate_b, a_rows, b_rows = best
            panels.append({
                "item": item,
                "pos": pos,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "a_ids": [row["instance_id"] for row in a_rows],
                "b_ids": [row["instance_id"] for row in b_rows],
                "strength": best[0][0],
            })

    print(f"eligible={len(panels)}")
    for panel in sorted(panels, key=lambda row: (-row["strength"], row["item"])):
        print(panel)


if __name__ == "__main__":
    main()
