"""Secondary object-label sensitivity, specified before opening confirmation outputs."""
import re
import unicodedata

import content_origin_material as M

EXCEPTIONS = {"peaches":"peach","cherries":"cherry","berries":"berry","mangoes":"mango",
              "geese":"goose","wolves":"wolf","foxes":"fox","horses":"horse","brushes":"brush",
              "benches":"bench","bicycles":"bicycle","scarves":"scarf","scissors":"scissors"}
EXTRA = {"berries":["浆果"],"pears":["梨子"],"violins":["提琴"],"notebooks":["笔记簿"],
         "bicycles":["脚踏车"],"pencils":["鉛筆"],"bottles":["玻璃樽"],"kettles":["水煲"]}


def object_origin(text,row):
    text=unicodedata.normalize("NFKC",text).lower()
    spans=[]
    for key,word in (("task",row["task_word"]),("archive",row["archive_word"])):
        singular=EXCEPTIONS.get(word,word.removesuffix("s"))
        for match in re.finditer(r"\b(?:"+word+"|"+singular+r")\b",text):
            spans.append((*match.span(),key))
        for alias in [*M.LEX[word],*EXTRA.get(word,())]:
            for match in re.finditer(re.escape(alias),text):
                spans.append((*match.span(),key))
    active={kind for a,b,kind in spans
            if not any(c<=a and b<=d and (c<a or b<d) for c,d,_ in spans)}
    return next(iter(active)) if len(active)==1 else ("both" if active else "other_or_missing")
