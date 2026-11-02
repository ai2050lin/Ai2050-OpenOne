# -*- coding: utf-8 -*-
"""FU-v2 材料：32 个新事件（与 FU-v1 不重叠）× 风格（日常 / 正式）× 逻辑（因果 / 转折）× 语法槽位（动词 / 名词 / 形容词），英文与中文。
风格操纵比 FU-v1 更强：正式风格同时更换前提措辞与连接词。
  英文：日常「{前提}, so|but the {主语}」；正式「{前提}; consequently|nevertheless, the {主语}」
        槽位：动词 = 主语后；名词 = 主语 + "'"；形容词 = 主语 + " were"
  中文：日常「{前提}，所以|但是{主语}」；正式「{前提}；因此|然而，{主语}」
        槽位：动词 = 主语后；名词 = 主语 + "的"；形容词 = 主语 + "都很"
"""
from __future__ import annotations

ITEMS = [
    # (英文主语, 英文日常, 英文正式, 中文主语, 中文日常, 中文正式)
    ('workers', 'The boss cut everyone\'s pay', 'Management implemented an across-the-board reduction in remuneration',
     '工人们', '老板给大家降了工资', '管理层对全体员工实施了薪酬削减'),
    ('farmers', 'The locusts ate the crops', 'A locust infestation devastated the agricultural yield',
     '农民们', '蝗虫把庄稼吃光了', '蝗灾对农作物造成了毁灭性损失'),
    ('students', 'The teacher gave a surprise quiz', 'The instructor administered an unannounced assessment',
     '学生们', '老师突然搞了个小测验', '任课教师组织了一次未经通知的测验'),
    ('kids', 'The ice cream truck came by', 'A mobile confectionery vendor entered the vicinity',
     '孩子们', '卖冰淇淋的车开过来了', '流动冷饮售卖车辆驶入该区域'),
    ('residents', 'The power went out', 'An interruption of the electrical supply occurred',
     '居民们', '突然停电了', '该地区发生了电力供应中断'),
    ('travelers', 'The flight got canceled', 'The scheduled flight was officially cancelled',
     '旅客们', '航班被取消了', '该航班已被正式取消'),
    ('employees', 'The office moved downtown', 'The company relocated its headquarters to the city center',
     '员工们', '公司搬到市中心去了', '公司总部已迁至市中心'),
    ('parents', 'The school lunch got worse', 'The nutritional quality of school meals deteriorated',
     '家长们', '学校的午饭越来越难吃', '学校午餐的营养质量明显下降'),
    ('fans', 'Their team lost the final', 'The team was defeated in the championship final',
     '球迷们', '他们的球队输了决赛', '该球队在决赛中落败'),
    ('campers', 'A bear walked into the camp', 'A bear entered the encampment',
     '露营者们', '一头熊走进了营地', '一头熊闯入了营地范围'),
    ('bakers', 'The flour ran out', 'The supply of flour was exhausted',
     '面包师们', '面粉用完了', '面粉库存已经耗尽'),
    ('guests', 'The hotel lost their bookings', 'The hotel failed to retain the reservation records',
     '客人们', '酒店把他们的预订弄丢了', '酒店未能保存相关预订记录'),
    ('reporters', 'The minister refused to talk', 'The minister declined to provide any statement',
     '记者们', '部长不肯说话', '部长拒绝发表任何声明'),
    ('volunteers', 'The storm wrecked the town', 'The storm inflicted extensive damage upon the town',
     '志愿者们', '暴风雨把小镇毁了', '暴风雨对该镇造成了大范围破坏'),
    ('drivers', 'Gas prices doubled', 'Fuel prices increased twofold',
     '司机们', '油价翻了一倍', '燃油价格上涨了一倍'),
    ('artists', 'The gallery shut its doors', 'The gallery ceased its public operations',
     '艺术家们', '画廊关门了', '该画廊已停止对外营业'),
    ('cyclists', 'The bike lane was removed', 'The designated cycling lane was eliminated',
     '骑车的人们', '自行车道被拆了', '专用自行车道已被撤除'),
    ('coders', 'The server crashed at midnight', 'The server experienced a critical failure at midnight',
     '程序员们', '服务器半夜崩了', '服务器于午夜发生严重故障'),
    ('builders', 'The cement did not arrive', 'The cement delivery failed to materialize',
     '建筑工人们', '水泥没送到', '水泥未能按期送达'),
    ('dancers', 'The stage was slippery', 'The stage surface presented a slipping hazard',
     '舞者们', '舞台很滑', '舞台表面存在滑倒风险'),
    ('owners', 'The dog chewed the sofa', 'The canine inflicted damage upon the upholstered furniture',
     '主人们', '狗把沙发咬坏了', '犬只对布艺家具造成了损坏'),
    ('singers', 'The microphones stopped working', 'The audio equipment malfunctioned',
     '歌手们', '话筒坏了', '音响设备出现了故障'),
    ('chefs', 'The fish was not fresh', 'The fish failed to meet freshness standards',
     '厨师们', '鱼不新鲜', '鱼类原料未达到新鲜度标准'),
    ('tenants', 'The landlord raised the rent', 'The property owner increased the rental charges',
     '租客们', '房东涨房租了', '业主上调了租金'),
    ('commuters', 'The subway broke down', 'The metro system suffered a service disruption',
     '上班族们', '地铁坏了', '地铁系统发生运营中断'),
    ('patients', 'The clinic ran out of vaccines', 'The clinic exhausted its vaccine inventory',
     '病人们', '诊所的疫苗用完了', '该诊所的疫苗库存已经耗尽'),
    ('players', 'The referee made a bad call', 'The referee issued an erroneous decision',
     '球员们', '裁判判错了', '裁判作出了错误判罚'),
    ('scouts', 'The map was wrong', 'The map contained significant inaccuracies',
     '童子军们', '地图画错了', '地图存在重大错误'),
    ('viewers', 'The show was canceled after one season', 'The series was discontinued following its first season',
     '观众们', '这部剧播了一季就被砍了', '该剧在首季播出后即被停播'),
    ('writers', 'The editor hated the draft', 'The editor expressed strong disapproval of the manuscript',
     '作家们', '编辑很讨厌这份稿子', '编辑对该稿件表示强烈不满'),
    ('sellers', 'Nobody came to the market', 'The market attracted no customers',
     '摊贩们', '集市上一个人都没来', '集市未能吸引任何顾客'),
    ('gamers', 'The game update deleted their saves', 'The software update erased the saved progress data',
     '玩家们', '游戏更新把存档删了', '软件更新导致存档数据丢失'),
]
STYLES = ['plain', 'formal']
LOGIC = ['cause', 'contrast']
SLOTS = ['verb', 'noun', 'adj']
CONN = {'en': {0: ('so', 'but'), 1: ('consequently', 'nevertheless')}, 'zh': {0: ('所以', '但是'), 1: ('因此', '然而')}}
SLOT_SUFFIX = {'en': ['', "'", ' were'], 'zh': ['', '的', '都很']}


def prompt(item, lang, s, l, g):
    es, ep, ef, zs, zp, zf = item
    c = CONN[lang][s][l]
    if lang == 'en':
        prem = ep if s == 0 else ef
        head = f'{prem}, {c} the {es}' if s == 0 else f'{prem}; {c}, the {es}'
    else:
        prem = zp if s == 0 else zf
        head = f'{prem}，{c}{zs}' if s == 0 else f'{prem}；{c}，{zs}'
    return head + SLOT_SUFFIX[lang][g]


def cells():
    """格序 c = 6s + 3l + g（g ∈ 0, 1, 2）。"""
    return [(s, l, g) for s in (0, 1) for l in (0, 1) for g in (0, 1, 2)]
