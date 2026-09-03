#!/usr/bin/env python3
"""Exploratory inventory of directly licensed WordNet noun relations for C089."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/gpt5/result/phase602_three_track_semantics/source/WordNet-3.0/dict/data.noun"
INDEX = ROOT / "tests/gpt5/result/phase602_three_track_semantics/source/WordNet-3.0/dict/index.noun"
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer


def parse():
    synsets = {}
    with DATA.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(line) < 10 or not line[:8].isdigit():
                continue
            body, _, gloss = line.partition("|")
            fields = body.split()
            offset, lex_file, pos = fields[:3]
            count = int(fields[3], 16)
            cursor = 4
            words = []
            for _ in range(count):
                words.append(fields[cursor])
                cursor += 2
            pointer_count = int(fields[cursor])
            cursor += 1
            pointers = []
            for _ in range(pointer_count):
                symbol, target, target_pos, source_target = fields[cursor:cursor + 4]
                pointers.append((symbol, target, target_pos, source_target))
                cursor += 4
            synsets[offset] = {"offset": offset, "lex_file": lex_file, "words": words, "pointers": pointers, "gloss": gloss.strip()}
    return synsets


def valid(word):
    return word.isalpha() and word.islower() and 3 <= len(word) <= 18


def tagged_words():
    tagged = {}
    with INDEX.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0].isspace():
                continue
            fields = line.split()
            if len(fields) < 6:
                continue
            synset_count = int(fields[2])
            pointer_count = int(fields[3])
            tag_count_index = 4 + pointer_count + 1
            count = int(fields[tag_count_index])
            if count > 0:
                tagged[fields[0]] = count
    return tagged


def main():
    synsets = parse()
    tag_counts = tagged_words()
    tagged = set(tag_counts)
    senses = Counter(word for synset in synsets.values() for word in synset["words"] if valid(word))
    relation_symbols = {"hypernym": "@", "part_of": "#p", "member_of": "#m", "substance_of": "#s", "antonym": "!"}
    counts = Counter()
    examples = defaultdict(list)
    synonym_pairs = []
    for synset in synsets.values():
        mono = [word for word in synset["words"] if valid(word) and senses[word] == 1]
        synonym_pairs.extend((left, right, synset["offset"]) for left, right in combinations(mono, 2))
    counts["synonym"] = len(synonym_pairs)
    examples["synonym"] = synonym_pairs[:8]
    for name, symbol in relation_symbols.items():
        rows = []
        for synset in synsets.values():
            source_words = [word for word in synset["words"] if valid(word) and senses[word] == 1]
            if not source_words:
                continue
            for pointer, target_offset, target_pos, _ in synset["pointers"]:
                if pointer != symbol or target_pos != "n" or target_offset not in synsets:
                    continue
                target_words = [word for word in synsets[target_offset]["words"] if valid(word) and senses[word] == 1]
                for source in source_words:
                    for target in target_words:
                        if source != target:
                            rows.append((source, target, synset["offset"], target_offset))
        counts[name] = len(rows)
        examples[name] = rows[:8]
    print("synsets", len(synsets))
    print("counts", dict(counts))
    for name in ("synonym", *relation_symbols):
        print(name, examples[name])
    print("tagged_words", len(tagged))
    for name, symbol in relation_symbols.items():
        tagged_count = 0
        tagged_examples = []
        for synset in synsets.values():
            source_words = [word for word in synset["words"] if valid(word) and senses[word] == 1 and word in tagged]
            for pointer, target_offset, target_pos, _ in synset["pointers"]:
                if pointer == symbol and target_pos == "n" and target_offset in synsets:
                    target_words = [word for word in synsets[target_offset]["words"] if valid(word) and senses[word] == 1 and word in tagged]
                    tagged_count += len(source_words) * len(target_words)
                    tagged_examples.extend((source, target) for source in source_words for target in target_words)
        print("tagged", name, tagged_count, tagged_examples[:20])

    tok = tokenizer()
    token_ok = {
        word
        for word in senses
        if word in tagged and len(tok.encode(" " + word, add_special_tokens=False)) <= 3
    }
    candidates = {"synonym": []}
    for source, target, source_offset in synonym_pairs:
        if source in token_ok and target in token_ok:
            candidates["synonym"].append((source, target, source_offset, source_offset))
    for name, symbol in relation_symbols.items():
        rows = []
        for synset in synsets.values():
            source_words = [word for word in synset["words"] if word in token_ok and senses[word] == 1]
            for pointer, target_offset, target_pos, _ in synset["pointers"]:
                if pointer != symbol or target_pos != "n" or target_offset not in synsets:
                    continue
                target_words = [word for word in synsets[target_offset]["words"] if word in token_ok and senses[word] == 1]
                rows.extend((source, target, synset["offset"], target_offset) for source in source_words for target in target_words if source != target)
        candidates[name] = rows
    for name in ("synonym", "hypernym", "part_of", "antonym"):
        selected, used = [], set()
        ranked = sorted(
            set(candidates[name]),
            key=lambda row: (-min(tag_counts[row[0]], tag_counts[row[1]]), -sum((tag_counts[row[0]], tag_counts[row[1]])), row),
        )
        for row in ranked:
            if row[0] in used or row[1] in used:
                continue
            selected.append(row)
            used.update(row[:2])
        print(
            "token_filtered", name, len(candidates[name]), "disjoint", len(selected),
            [(row[0], tag_counts[row[0]], row[1], tag_counts[row[1]]) for row in selected[:40]],
        )


if __name__ == "__main__":
    main()
