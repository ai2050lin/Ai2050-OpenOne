#!/usr/bin/env python3
"""Small deterministic helpers shared by the C043 measurement stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def chat_ids(tokenizer: Any, system: str, prompt: str) -> list[int]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False)
    except TypeError:
        value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def locate_last_subsequence(values: list[int], needles: list[list[int]]) -> list[int]:
    matches: set[tuple[int, ...]] = set()
    for needle in needles:
        for start in range(len(values) - len(needle) + 1):
            if values[start:start + len(needle)] == needle:
                matches.add(tuple(range(start, start + len(needle))))
    if not matches:
        raise ValueError("no token subsequence matched")
    return list(sorted(matches, key=lambda item: (item[-1], len(item)))[-1])


def locate_word(tokenizer: Any, text: str, word: str) -> tuple[list[int], list[int]]:
    ids = [int(item) for item in tokenizer.encode(text, add_special_tokens=False)]
    needles = [[int(item) for item in tokenizer.encode(form, add_special_tokens=False)]
               for form in (word, " " + word)]
    return ids, locate_last_subsequence(ids, needles)

