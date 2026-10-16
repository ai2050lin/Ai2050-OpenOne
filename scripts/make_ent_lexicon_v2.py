# -*- coding: utf-8 -*-
"""ENT-XF 词表 v1 → v2 + 改动记录（Phase 250）。备选色清单 ALT / ALT2 / ALT3 即复核结论；v1 行为与自由回答只写入来源列。

    python scripts/make_ent_lexicon_v2.py   # 需要 ~/.ai2050_agent/tasks 下 v1 的 behavior.json 与自由回答 JSON
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'shared/dist_protocol/lexicon/ENT-XF_nouns.v1.tsv'
V2 = ROOT / 'shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv'
CH = ROOT / 'shared/dist_protocol/lexicon/ENT-XF_nouns.v2.changes.tsv'
TASKS = ['T-2d80e249bd', 'T-1d5127647f', 'T-6a5b12f049', 'T-06227755ac']

# 规则（写于 v2 重跑之前）：R1 只加入指称对象真实、常见的颜色（品种 / 成熟度 / 常见形态），不加入环境或联想色
# （海→蓝、植物→绿）与图示色；R2 每个名词至多 2 个备选色（v2 第二轮放宽），中英文相同；R3 对全部 210 个名词审阅，不只审失败项；
# R4 典型颜色在 11 色调色板之外时，允许调色板外的可接受答案（不进入任何对照集）。
ALT = {
    'fruit.grape': ('green', '绿葡萄（青提）是常见品种'),
    'fruit.mango': ('orange', '成熟果皮与果肉常为橙黄'),
    'fruit.coconut': ('green', '青椰（嫩椰）在售卖中常见，外果皮绿色'),
    'fruit.apricot': ('yellow', '杏多为黄橙色（「杏黄」）'),
    'fruit.plum': ('red', '红李是常见品种'),
    'fruit.fig': ('green', '绿皮无花果是常见品种'),
    'fruit.persimmon': ('red', '成熟柿子橙红，中文常称「红柿子」'),
    'fruit.loquat': ('orange', '枇杷果皮黄橙色'),
    'fruit.starfruit': ('green', '杨桃常以青黄未熟状态售卖'),
    'fruit.apple': ('green', '青苹果是常见品种'),
    'fruit.mulberry': ('black', '成熟桑葚紫黑'),
    'fruit.pomelo': ('green', '柚子果皮常为黄绿 / 绿色'),
    'fruit.passionfruit': ('yellow', '黄皮百香果是常见品种'),
    'fruit.gooseberry': ('red', '红醋栗与绿醋栗都常见'),
    'fruit.nectarine': ('yellow', '油桃果皮红黄相间'),
    'vegetable.pumpkin': ('yellow', '南瓜多为橙黄，中文常称黄色'),
    'vegetable.chili': ('green', '青椒 / 青辣椒常见'),
    'vegetable.artichoke': ('purple', '紫色洋蓟是常见品种'),
    'vegetable.potato': ('yellow', '土豆果肉黄色，中文常称黄色'),
    'vegetable.ginger': ('brown', '生姜外皮为褐色'),
    'vegetable.beet': ('purple', '甜菜根为紫红色'),
    'mammal.giraffe': ('brown', '毛色为黄底棕色斑块'),
    'mammal.lion': ('brown', '毛色为黄褐（tawny）'),
    'mammal.tiger': ('yellow', '中文常称「黄色」底黑纹'),
    'mammal.horse': ('black', '黑马是常见毛色之一'),
    'mammal.squirrel': ('gray', '灰松鼠是常见种'),
    'mammal.camel': ('yellow', '毛色沙黄 / 黄褐'),
    'mammal.fox': ('red', '赤狐（red fox）是最常见的狐'),
    'bird.sparrow': ('gray', '麻雀羽色褐与灰相间'),
    'bird.owl': ('gray', '许多猫头鹰为灰褐色'),
    'bird.peacock': ('green', '孔雀羽色蓝绿，绿孔雀是常见种'),
    'bird.falcon': ('gray', '游隼背部灰色'),
    'bird.seagull': ('gray', '海鸥背羽灰色'),
    'aquatic.sea-turtle': ('brown', '海龟背甲多为褐色'),
    'aquatic.octopus': ('brown', '活章鱼多为褐色（红色多见于烹熟）'),
    'aquatic.sea-cucumber': ('brown', '海参多为黑褐色'),
    'aquatic.piranha': ('red', '红腹食人鱼是最常见种'),
    'aquatic.jellyfish': ('transparent', '水母通常透明（调色板外，R4）'),
    'plant.lotus': ('white', '白荷花常见'),
    'plant.chrysanthemum': ('white', '白菊花常见'),
    'plant.hyacinth': ('blue', '蓝色风信子是常见品种'),
    'plant.maple': ('green', '枫叶一年中多数时候为绿色'),
    'plant.camellia': ('pink', '粉色茶花常见'),
    'food.cheese': ('white', '许多奶酪为白色'),
    'food.soy-sauce': ('brown', '酱油为深褐色'),
    'food.cola': ('brown', '可乐为深棕色'),
}

# 第二轮（自由回答校准后）：模型自由回答提出、且符合 R1（现实中常见）的颜色；R2 放宽为至多 2 个备选色
ALT2 = {
    'aquatic.abalone': ('brown', '鲍鱼壳多为深褐色'),
    'aquatic.catfish': ('brown', '鲶鱼体色多为深褐 / 灰褐'),
    'aquatic.eel': ('brown', '鳗鱼背部多为橄榄褐色'),
    'aquatic.manatee': ('brown', '海牛体色灰褐'),
    'aquatic.oyster': ('white', '牡蛎肉与壳内侧为白色'),
    'aquatic.perch': ('green', '黄鲈 / 河鲈体色黄绿'),
    'aquatic.salamander': ('brown', '许多蝾螈为褐色'),
    'aquatic.trout': ('green', '鳟鱼背部多为橄榄绿'),
    'aquatic.walrus': ('gray', '海象体色常描述为灰褐'),
    'aquatic.whale': ('black', '虎鲸、露脊鲸、座头鲸等常见鲸为黑色'),
    'bird.ostrich': ('brown', '雌鸵鸟与幼鸟为褐色'),
    'bird.turtledove': ('brown', '斑鸠羽色灰褐'),
    'food.caviar': ('red', '红鱼子（鲑鱼籽）常见'),
    'food.century-egg': ('brown', '皮蛋蛋白为深褐色半透明'),
    'food.peanut': ('red', '花生仁外皮（花生衣）为红色'),
    'food.pudding': ('white', '奶布丁为白色'),
    'food.wasabi': ('yellow', '中文「芥末」也指黄芥末'),
    'fruit.grape': ('red', '红提 / 红葡萄常见'),
    'fruit.lemon': ('green', '青柠檬常见'),
    'fruit.mango': ('red', '红芒（红皮芒果）常见'),
    'fruit.plum': ('green', '青李常见'),
    'mammal.armadillo': ('brown', '九带犰狳为灰褐色'),
    'mammal.badger': ('brown', '美洲獾为褐色'),
    'mammal.yak': ('brown', '野牦牛为深褐色'),
    'plant.peach-blossom': ('red', '红桃花（碧桃）常见'),
    'vegetable.carrot': ('red', '中文常称「红萝卜」，红色胡萝卜品种'),
    'vegetable.potato': ('white', '土豆果肉多为白色'),
}
# 第三轮（最后一轮，选定问答式提示后）：问答式提示的自由回答提出、且符合 R1 的颜色；此后不再迭代
ALT3 = {
    'mammal.koala': [('brown', '考拉毛色灰褐')],
    'bird.vulture': [('brown', '兀鹫等常见秃鹫为褐色')],
    'aquatic.mussel': [('brown', '贻贝壳常为深褐色')],
    'aquatic.sturgeon': [('brown', '鲟鱼背部多为灰褐 / 深褐')],
    'aquatic.crab': [('orange', '许多蟹（及煮熟的蟹）为橙红色')],
    'food.honey': [('amber', '蜂蜜常描述为琥珀色（调色板外，R4）')],
    'food.beer': [('amber', '啤酒常描述为琥珀色（调色板外，R4）'), ('brown', '黑啤 / 深色啤酒常见')],
    'food.curry': [('brown', '日式咖喱为褐色')],
    'plant.oak': [('brown', '橡树树干与木材为褐色')],
}
PALETTE = {'red', 'yellow', 'green', 'blue', 'white', 'black', 'brown', 'orange', 'purple', 'pink', 'gray', 'transparent',
           'amber'}


def v1_behavior():
    per = defaultdict(dict)
    for t in TASKS:
        p = Path(os.path.expanduser(f'~/.ai2050_agent/tasks/{t}/outputs/behavior.json'))
        for it in json.loads(p.read_text(encoding='utf-8'))['items']:
            fr, noun = it['id'].split(':')[1:]
            lang, _, task, _ = fr.split('-')
            per[(noun, task)][lang] = (it['correct'], it.get('best_contrast'), round(it.get('margin', 0), 2))
    return per


def main():
    beh = v1_behavior()
    free = {}
    fp = ROOT / 'research/family_runs/entangle/ENT-XF-v2-freeresponse-qwen3-4b.json'
    for it in json.loads(fp.read_text(encoding='utf-8'))['items']:
        if it['task'] == 'color':
            sl = ' '.join(f'{k}×{v}' for k, v in sorted(it['sample_labels'].items(), key=lambda kv: -kv[1]))
            free.setdefault(it['id'], {})[it['lang']] = f"{it['greedy_label']} | {sl}"
    lines = V1.read_text(encoding='utf-8').splitlines()
    out = ['# ENT-XF 名词词表 v2：id\tcat\tcat_zh\tsuper\tzh\ten\tcolor\tcolor_alt\tsize',
           '# 来源：v1（作者 Claude 依常识编写，2026-10-16）+ v2 复核（2026-10-16，Phase 250）：color = 主颜色（分析标签），'
           'color_alt = 至多 2 个备选可接受颜色（第 1 个来自现实复核，第 2 个来自自由回答提出 + 现实复核）（只用于行为门的可接受答案，不改变分析标签；transparent 为调色板外答案）。',
           '# v2 复核规则与逐项理由见 ENT-XF_nouns.v2.changes.tsv；名词、类别、大小标签与 v1 相同。',
           '# 筛选：名称不含颜色词；去掉近重复词、颜色或大小有明显歧义的词；见 scripts/build_ent_grammar.py 的检查。']
    ch = ['# ENT-XF 词表 v1 → v2 改动记录（Phase 250）。规则 R1–R4 见 scripts/build_ent_grammar.py 文档。',
          '# v1_color_zh / v1_color_en = v1 运行（TG-ENT-XF v1，design bc3567a97921）的颜色强制选择：通过 / 失败（最强对照，间隔）。',
          '# 这些行为信息只作来源说明：改动依据是 reason 列中的现实依据；模型偏好但无现实依据的颜色（如 天鹅→蓝）不加入。',
          '# free_zh / free_en = Qwen3-4B 对 v1 式提示「{X}的颜色通常是」/「The typical color of the {E} is」的自由回答（贪心标签 | 5 次采样的标签计数；第三轮另用问答式提示，见 *-candB.json），见 '
          'research/family_runs/entangle/ENT-XF-v2-freeresponse-qwen3-4b.json；只用于提出候选颜色，入选与否按 R1 的现实依据。',
          'id\tzh\tfield\tv1\tv2\tv1_color_zh\tv1_color_en\tfree_zh\tfree_en\tsource\treason']
    seen = set()
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        nid, cat, cat_zh, sup, zh, en, color, size = line.split('\t')
        alts = [(*ALT[nid], '现实复核')] if nid in ALT else []
        if nid in ALT2:
            alts.append((*ALT2[nid], '自由回答提出 + 现实复核'))
        for a, w in ALT3.get(nid, []):
            alts.append((a, w, '问答式提示自由回答提出 + 现实复核'))
        assert len(alts) <= 2 and all(a in PALETTE and a != color for a, _, _ in alts), nid
        assert len({a for a, _, _ in alts}) == len(alts), nid
        seen.add(nid)
        fr = free.get(nid, {})
        out.append('\t'.join([nid, cat, cat_zh, sup, zh, en, color, ','.join(a for a, _, _ in alts), size]))
        b = beh.get((nid, 'color'), {})
        fmt = lambda l: ('通过' if b[l][0] else f'失败（{b[l][1].strip()}，{b[l][2]}）') if l in b else '-'
        for a, why, src in alts:
            ch.append('\t'.join([nid, zh, 'color_alt', '', a, fmt('zh'), fmt('en'), fr.get('zh', '-'), fr.get('en', '-'),
                                 src, why]))
    assert set(ALT) | set(ALT2) | set(ALT3) <= seen, (set(ALT) | set(ALT2) | set(ALT3)) - seen
    V2.write_text('\n'.join(out) + '\n', encoding='utf-8')
    CH.write_text('\n'.join(ch) + '\n', encoding='utf-8')
    print(len(out) - 4, 'nouns;', len(ch) - 5, 'changes')


if __name__ == '__main__':
    main()
