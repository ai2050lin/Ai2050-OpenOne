# -*- coding: utf-8 -*-
"""Q8-A 预探 2：候选词表的分词形态覆盖率。"""
import numpy as np
from transformers import AutoTokenizer
from wordfreq import zipf_frequency as zf

tok = AutoTokenizer.from_pretrained('D:/develop/model/Qwen/Qwen3-4B')

COLOR = ['红', '大红', '朱红', '绯红', '桃红', '粉红', '洋红', '玫瑰红', '酒红', '枣红', '橘红',
         '橙', '橘黄', '金黄', '黄', '柠檬黄', '米黄', '土黄', '草绿', '嫩绿', '翠绿', '绿',
         '墨绿', '橄榄绿', '碧绿', '湖蓝', '天蓝', '浅蓝', '宝蓝', '蓝', '藏青', '靛蓝', '青',
         '青绿', '紫', '紫罗兰', '丁香紫', '葡萄紫', '粉', '棕', '咖啡色', '巧克力色', '卡其色',
         '驼色', '灰', '银灰', '铁灰', '烟灰', '黑', '雪白', '白', '米色']

FRUIT = ['苹果', '香蕉', '橙子', '橘子', '柑', '柠檬', '柚子', '葡萄', '草莓', '蓝莓', '树莓',
         '西瓜', '甜瓜', '哈密瓜', '桃子', '梨', '李子', '杏', '樱桃', '芒果', '菠萝', '木瓜',
         '荔枝', '龙眼', '榴莲', '山竹', '火龙果', '猕猴桃', '柿子', '石榴', '无花果', '杨梅',
         '桑葚', '枇杷', '山楂', '椰子', '牛油果', '百香果', '杨桃', '莲雾', '枣', '橄榄',
         '板栗', '核桃', '榛子', '腰果', '花生', '番石榴', '蜜瓜', '蔓越莓']

RARE = ['龘', '靐', '齉', '爨', '麤', '鱻', '鑫', '犇', '羴', '猋', '骉', '毳', '掱', '嫑',
        '兲', '巭', '勥', '埊', '玊', '烎', '氼', '嘦', '圕', '孖', '冇', '乜', '冚', '蠱',
        '彧', '翀', '仝', '亓', '甪', '氍', '觳', '黼', '黻', '鼗', '龠', '龢', '籴', '粜',
        '耷', '耨', '薅', '搴', '攴', '殳', '厍', '卺', '耄', '耋', '饕', '餮', '魑', '魅']

RAD = {
    '氵': '江河湖海洋波浪流清洗泳溪池滩湾潮',
    '木': '松柏杨柳枫桃梅桂桐榆槐枝根树林桥楼',
    '火': '烧烤炒炸煮蒸烙炼燃焰灯炉炮燥',
    '讠': '说话讲谈论议评谢请谁词语读记识训访询',
    '忄': '情恨惧怕悔忙慢懂慌怖惊悦愁恰惭愧',
    '石': '硬碰碗破碎砖砂码磁确',
    '虫': '蚊蝇蝶蜂蚁蛇虾蟹蜘蛛蜜蜡',
    '鸟': '鸡鸭鹅鹤鸦鸽鹰燕鹏鹊',
}


def report(name, ws):
    n1 = [w for w in ws if len(tok.encode(w, add_special_tokens=False)) == 1]
    print('%-6s total=%d single=%d (%.0f%%)' % (name, len(ws), len(n1), 100 * len(n1) / len(ws)))
    multi = [w for w in ws if w not in n1]
    print('   single:', ' '.join(n1))
    print('   multi :', ' '.join(multi))


report('COLOR', COLOR)
report('FRUIT', FRUIT)
report('RARE', RARE)
for k, v in RAD.items():
    report('RAD' + k, list(v))

# 生僻字频率分布
print('rare zipf:', [(w, zf(w, 'zh')) for w in RARE[:12]])
print('bytes of 龘:', list('龘'.encode('utf-8')))
print('byte tokens:', [tok.decode([i]) for i in tok.encode('龘', add_special_tokens=False)])
print('cp 龘 靐 齉:', [hex(ord(c)) for c in '龘靐齉'])
