/* 语言图谱 · 词汇关系环图的数据：只读公开的外部词表，不含任何模型输出。
   - ENT-XF 名词词表 v2（210 词）：类别、大类、人工标注的常识颜色、备选颜色、大小（S / L）
   - EA-v2 新实体（197 词）：类别、大类
   - MA-H1 discovery 划分（84 词）：类别；confirm 与 lockbox 封存在节点本地，这里不读 */
import entxf from '../../../../../shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv?raw';
import eav2 from '../../../../../shared/dist_protocol/lexicon/EA-v2_new_entities.tsv?raw';
import mah1 from '../../../../../research/mainline/H1/discovery.tsv?raw';

export const CAT_ZH = {
  mammal: '哺乳动物', bird: '鸟类', aquatic: '水生动物', insect: '昆虫', reptile: '爬行与两栖',
  fruit: '水果', vegetable: '蔬菜', plant: '花草树木', food: '食品',
  tool: '工具', instrument: '乐器', vehicle: '交通工具', kitchen: '厨具', stationery: '文具', household: '日用品', clothing: '服饰', furniture: '家具', sports: '运动器材',
  mineral: '矿物',
};
export const SUPERS = [['animal', '动物'], ['plant', '植物'], ['food', '食品'], ['artifact', '人造物'], ['nature', '矿物']];
const SUPER_OF = { insect: 'animal', reptile: 'animal', mineral: 'nature' };
export const COLOR_ZH = { red: '红', yellow: '黄', green: '绿', blue: '蓝', white: '白', black: '黑', brown: '棕', orange: '橙', purple: '紫', pink: '粉', gray: '灰', amber: '琥珀', transparent: '透明' };
export const COLOR_HEX = { red: '#d03b3b', yellow: '#d9b100', green: '#2f9a4a', blue: '#2a6fd6', white: '#9aa3ad', black: '#22262b', brown: '#8a5a2b', orange: '#e47a1c', purple: '#7a4bb5', pink: '#e06aa0', gray: '#7d8590', amber: '#c98a00', transparent: '#b9c4d2' };
export const SIZE_ZH = { S: '比成年人小', L: '比成年人大' };

const rows = text => text.split(/\r?\n/).filter(line => line && !line.startsWith('#')).map(line => line.split('\t'));

function build() {
  const words = [];
  for (const [id, cat, , sup, zh, en, color, alt, size] of rows(entxf)) {
    words.push({ id, cat, sup, zh, en, source: 'ENT-XF', color: color || '', alt: alt ? alt.split(',') : [], size: size || '' });
  }
  for (const [id, cat, sup, zh, en] of rows(eav2)) words.push({ id, cat, sup, zh, en, source: 'EA-v2', color: '', alt: [], size: '' });
  for (const [id, cat, zh, en] of rows(mah1).slice(1)) {
    words.push({ id, cat, sup: SUPER_OF[cat] || 'artifact', zh, en, source: 'MA-H1 discovery', color: '', alt: [], size: '' });
  }
  const supOrder = Object.fromEntries(SUPERS.map(([k], i) => [k, i]));
  const catOrder = Object.fromEntries(Object.keys(CAT_ZH).map((k, i) => [k, i]));
  return words.sort((a, b) => supOrder[a.sup] - supOrder[b.sup] || catOrder[a.cat] - catOrder[b.cat] || a.zh.localeCompare(b.zh, 'zh-CN'));
}

export const WORDS = build();
export const SOURCES = [...new Set(WORDS.map(w => w.source))].map(s => [s, WORDS.filter(w => w.source === s).length]);
