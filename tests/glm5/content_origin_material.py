"""Original crossed archive sources. The decoder never sees the scoring lexicon."""
from itertools import product
from random import Random
import ast
import hashlib
import json
import re
import unicodedata

SIZES = {"pilot": 32, "discovery": 64, "confirm": 256, "lockbox": 128}
WORDS = {
    "pilot": "pears:梨 peaches:桃 apples:苹果 oranges:橙/桔/橘 lemons:柠檬 grapes:葡萄 cherries:樱桃 melons:瓜 berries:莓 bananas:香蕉 plums:李子/李 mangoes:芒果",
    "discovery": "cats:猫 dogs:狗/犬 rabbits:兔 horses:马 ducks:鸭 geese:鹅 wolves:狼 foxes:狐 bears:熊 goats:山羊/羊 cows:牛 birds:鸟 frogs:蛙 turtles:龟 lions:狮 tigers:虎",
    "confirm": "umbrellas:伞 candles:蜡烛/烛子 bicycles:自行车/单车 pillows:枕 lamps:灯 mirrors:镜 spoons:勺/匙 forks:叉 pencils:铅笔 notebooks:笔记本/记事本 bottles:瓶 baskets:篮 drums:鼓 scarves:围巾/丝巾 scissors:剪刀 socks:袜 gloves:手套 brushes:刷 benches:长椅/板凳/长凳 clocks:钟 ladders:梯 kettles:壶 magnets:磁铁/磁石 violins:小提琴 plates:盘/碟 towels:毛巾/浴巾",
    "lockbox": "helmets:头盔/安全帽 buttons:纽扣/钮扣/扣子 radios:收音机 suitcases:行李箱 boots:靴 belts:腰带/皮带 wallets:钱包 flutes:笛 guitars:吉他 carpets:地毯 curtains:帘 cameras:相机/照相机 ribbons:缎带/丝带 combs:梳 blankets:毯 buckets:桶",
}
FRAMES = [
    ("Archive entry", "Initial count", "Revised count", "Use the {source} record."),
    ("Archived example", "First inventory", "Revised inventory", "Report the {source} record."),
    ("Archive excerpt", "Original estimate", "Revised estimate", "Select the {source} record."),
    ("Archived sample", "Initial survey", "Revised survey", "Take both details from the {source} record."),
    ("Archived illustration", "First tally", "Revised tally", "Refer to the {source} record."),
]
FRAMES_BY_SPLIT = {"pilot": (0, 1), "discovery": (0, 1), "confirm": (2, 3), "lockbox": (4,)}
GUIDE = "Answer in Chinese. Name the object and give its count in square brackets. Use one Python print call with one message string. No other text."
LEX = {w: zh.split("/") for pool in WORDS.values() for entry in pool.split() for w, zh in [entry.split(":")]}
DIGITS = "零一二三四五六七八九"
EN_NUM = "zero one two three four five six seven eight nine".split()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def families(tok, split):
    rng = Random(298110 + list(SIZES).index(split))
    words = [v.split(":")[0] for v in WORDS[split].split()]
    bins = {}
    for word in words:
        bins.setdefault(len(tok.encode(" " + word, add_special_tokens=False)), []).append(word)
    groups = [g for g in bins.values() if len(g) >= 2]
    result, used = [], set()
    while len(result) < SIZES[split]:
        a, b = rng.sample(rng.choice(groups), 2)
        current = rng.choice([w for w in words if w not in (a, b)])
        nums = tuple(rng.sample(range(2, 10), 4))
        key = (current, *sorted((a, b)), *nums)
        if key in used:
            continue
        used.add(key)
        i = len(result)
        result.append(dict(family=digest([split, key])[:18], index=i, current=current,
                           archives=[a, b], numbers=nums, early=(i % 4 < 2), brackets=bool(i % 2)))
    return result


def render(tok, fam, frame, style, payload, requested, neutral=False):
    head, old, revised, choice = FRAMES[frame]
    archive_word, task_word = fam["archives"][payload], fam["current"]
    old_n, new_n, *archive_nums = fam["numbers"]
    archive_n = archive_nums[payload]
    n = f"[{archive_n}]" if fam["brackets"] else str(archive_n)
    sample = f'| print(" {n} {archive_word}")' if style else f"| We saw {n} {archive_word}."
    archive = head + ": " + sample
    facts = f"{old}: {old_n} {task_word}.\n{revised}: {new_n} {task_word}."
    sections = [archive, facts] if fam["early"] else [facts, archive]
    if neutral:
        sections = [archive] if requested else [facts]
    source = ("revised", "archived")[requested]
    body = "\n".join([*sections, choice.format(source=source), GUIDE])
    prompt = tok.apply_chat_template([dict(role="user", content=body)], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    def span(piece, start=0):
        begin = prompt.index(piece, start)
        end = begin + len(piece)
        return [i for i,(a,b) in enumerate(enc.offset_mapping) if a < end and b > begin]
    spans = {"request": span(source, prompt.index(choice.split("{")[0], prompt.index(sections[-1]) + len(sections[-1])) )}
    if not neutral or requested:
        spans["archive"] = span(sample)
        spans["payload"] = span(f"{n} {archive_word}", prompt.index(sample))
    if not neutral or not requested:
        spans["current"] = span(f"{new_n} {task_word}", prompt.index(revised + ":"))
    return dict(id=f"{fam['family']}-{frame}{style}{payload}{requested}{int(neutral)}",
                family=fam["family"], frame=frame, style=style, payload=payload, requested=requested,
                neutral=neutral, early=fam["early"], brackets=fam["brackets"], prompt=prompt,
                ids=list(enc.input_ids), spans=spans, task_word=task_word, archive_word=archive_word,
                old_n=old_n, new_n=new_n, archive_n=archive_n)


def build(tok, split):
    rows = []
    for fam in families(tok, split):
        for frame in FRAMES_BY_SPLIT[split]:
            if split == "pilot":
                rows.extend(render(tok, fam, frame, s, fam["index"] % 2, r, True) for s,r in product(range(2), repeat=2))
                continue
            panel = {key: render(tok, fam, frame, *key) for key in product(range(2), repeat=3)}
            for key, row in panel.items():
                for axis, region in enumerate(("archive", "payload", "request")):
                    other = panel[tuple((v ^ 1) if i == axis else v for i,v in enumerate(key))]
                    assert row["spans"] == other["spans"], (key, row["spans"],other["spans"])
                    changed = {i for i,(a,b) in enumerate(zip(row["ids"], other["ids"], strict=True)) if a != b}
                    assert changed and changed <= set(row["spans"][region]), (axis,changed,row["spans"])
                    if axis == 0:
                        assert not changed.intersection(row["spans"]["payload"])
            rows.extend(panel.values())
    return rows


def string_value(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return string_value(node.left) + string_value(node.right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str"
            and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is int):
        return str(node.args[0].value)
    raise ValueError


def score(text, row, ended):
    message, py = text, False
    try:
        root = ast.parse(text.strip(), mode="eval").body
        if (isinstance(root, ast.Call) and isinstance(root.func, ast.Name) and root.func.id == "print"
                and len(root.args) == 1 and not root.keywords):
            message = string_value(root.args[0])
            py = True
    except (SyntaxError, ValueError, RecursionError):
        pass
    message = unicodedata.normalize("NFKC", message)
    def has_object(word):
        return any(s in message for s in LEX[word]) or bool(re.search(r"\b" + re.escape(word) + r"\b",message,re.I))
    objs = [k for k, word in (("archive",row["archive_word"]),("task",row["task_word"])) if has_object(word)]
    # Preserve extra or contradictory numeric values, including numbers outside the task range.
    numbers = set()
    for value in re.findall(r"\d+|[零一二两兩三四五六七八九十百千万]+|\b(?:"+"|".join(EN_NUM)+r")\b",message.lower()):
        if value.isdecimal():
            numbers.add(int(value))
        elif value in EN_NUM:
            numbers.add(EN_NUM.index(value))
        else:
            value = value.replace("两","二").replace("兩","二")
            numbers.add(DIGITS.index(value) if len(value)==1 and value in DIGITS else value)
    lookup = {row["archive_n"]:"archive",row["new_n"]:"revised",row["old_n"]:"old"}
    num = next(iter(numbers)) if len(numbers)==1 else None
    origin = lookup.get(num,"other") if numbers else "missing"
    if len(numbers)>1:
        origin="multiple"
    obj = objs[0] if len(objs)==1 else ("both" if objs else "other_or_missing")
    target_obj, target_number = ("archive","archive") if row["requested"] else ("task","revised")
    source_correct = obj == target_obj and origin == target_number
    gold = row["archive_n"] if row["requested"] else row["new_n"]
    square = bool(re.search(r"\[\s*(?:"+str(gold)+"|"+DIGITS[gold]+("|两|兩" if gold==2 else "")+r")\s*\]",message))
    chinese = bool(re.search(r"[\u3400-\u9fff]", message)) and not re.search("[a-zA-Z]",message)
    return dict(obj=obj, num=origin, numbers=sorted(map(str,numbers)), archive_both=obj=="archive" and origin=="archive",
                correct=source_correct, python=py, chinese=bool(chinese), square=square,
                joint=bool(source_correct and py and chinese and square and ended))
