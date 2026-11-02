# -*- coding: utf-8 -*-
"""FU-v1 材料：40 个事件 × 风格（日常 / 正式）× 逻辑（so / but）× 语法槽位（动词 / 名词）的 2×2×2 因子设计（英文）。
提示 = 前提句（风格）+ ", " + 连接词（逻辑）+ " the " + 主语复数名词 + （语法：无 → 下一词是动词；"'" → 所有格之后下一词是名词）。
例：The road was icy, so the drivers ___ / Road surfaces exhibited substantial ice accumulation, but the drivers' ___
每个事件的四种前提共用同一主语；日常与正式前提描述同一事实。"""
from __future__ import annotations

import hashlib

ITEMS = [
    ('drivers', 'The road was icy', 'The roadway surface exhibited substantial ice accumulation'),
    ('students', 'The exam was very hard', 'The examination was of considerable difficulty'),
    ('farmers', 'It did not rain all summer', 'Precipitation was absent throughout the summer season'),
    ('workers', 'The factory closed down', 'The manufacturing facility ceased operations'),
    ('players', 'The coach yelled at the team', 'The coach addressed the team in a raised and critical manner'),
    ('tourists', 'The museum was packed', 'The museum was operating at maximum visitor capacity'),
    ('patients', 'The new drug worked well', 'The novel pharmaceutical agent demonstrated substantial efficacy'),
    ('neighbors', 'The music was really loud', 'The acoustic output was of exceptionally high volume'),
    ('investors', 'The stock price crashed', 'The equity valuation underwent a precipitous decline'),
    ('children', 'The park was full of snow', 'The park was covered by an extensive layer of snow'),
    ('runners', 'It was extremely hot outside', 'Ambient temperatures were exceptionally elevated'),
    ('voters', 'The election was close', 'The electoral outcome was narrowly contested'),
    ('customers', 'The shop raised its prices', 'The retailer implemented a general price increase'),
    ('soldiers', 'The bridge was destroyed', 'The bridge structure had been rendered inoperable'),
    ('fishermen', 'The sea was rough', 'Sea conditions were characterized by severe turbulence'),
    ('pilots', 'The fog was thick', 'Visibility was severely restricted by dense fog'),
    ('teachers', 'The school cut its budget', 'The institution implemented substantial budgetary reductions'),
    ('hikers', 'The trail was muddy', 'The trail surface was saturated and unstable'),
    ('readers', 'The book was boring', 'The volume was widely regarded as lacking in interest'),
    ('engineers', 'The test failed again', 'The evaluation procedure was unsuccessful on a repeated occasion'),
    ('passengers', 'The train was late', 'The rail service experienced a significant delay'),
    ('scientists', 'The data looked strange', 'The empirical observations exhibited anomalous characteristics'),
    ('cooks', 'The oven broke', 'The oven suffered a mechanical malfunction'),
    ('fans', 'The concert was canceled', 'The scheduled performance was officially withdrawn'),
    ('villagers', 'The river flooded', 'The river exceeded its banks and inundated the surrounding area'),
    ('doctors', 'The flu spread fast', 'The influenza outbreak propagated at a rapid rate'),
    ('swimmers', 'The water was freezing', 'The water temperature approached the point of freezing'),
    ('buyers', 'The house was cheap', 'The property was offered at a notably low valuation'),
    ('climbers', 'The storm came early', 'The storm system arrived ahead of forecasts'),
    ('lawyers', 'The judge was angry', 'The presiding judge expressed evident displeasure'),
    ('gardeners', 'The soil was dry', 'The soil exhibited a pronounced moisture deficit'),
    ('musicians', 'The hall was empty', 'The auditorium was devoid of an audience'),
    ('shoppers', 'The mall was closed', 'The shopping center was not open to the public'),
    ('birds', 'The winter was long', 'The winter season was of unusual duration'),
    ('miners', 'The tunnel collapsed', 'The tunnel underwent a structural collapse'),
    ('actors', 'The script was bad', 'The screenplay was of deficient quality'),
    ('citizens', 'The taxes went up', 'The tax burden was substantially increased'),
    ('sailors', 'The wind stopped', 'Wind activity ceased entirely'),
    ('nurses', 'The hospital was short of staff', 'The hospital experienced a significant personnel shortage'),
    ('athletes', 'The stadium was flooded', 'The stadium was affected by extensive flooding'),
]
STYLES = ['plain', 'formal']
LOGIC = ['so', 'but']
SLOTS = ['verb', 'noun']


def prompt(item, s, l, g):
    subj, plain, formal = item
    prem = plain if s == 0 else formal
    return f"{prem}, {LOGIC[l]} the {subj}" + ("'" if g == 1 else '')


def cells():
    """格序 c = 4s + 2l + g。"""
    return [(c >> 2 & 1, c >> 1 & 1, c & 1) for c in range(8)]


def split():
    """按事件确定性切分：发现 24 / 确认 16。"""
    order = sorted(range(len(ITEMS)), key=lambda i: hashlib.sha256(f'FU-v1|{ITEMS[i][0]}'.encode()).hexdigest())
    return {'conf': sorted(order[:16]), 'disc': sorted(order[16:])}
