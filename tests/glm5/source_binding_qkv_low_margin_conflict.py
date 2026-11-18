"""A third frozen cohort with two conflicting versions of the same named list."""
import argparse
from pathlib import Path

import source_binding_qkv_low_margin as experiment
import source_binding_qkv_low_margin_analysis as analysis
from source_binding_design import sha


def configure():
    experiment.OUT = experiment.OUT.parent / 'low-margin-conflict'
    experiment.SEED += 2
    experiment.FAMILIES = 192
    experiment.CHUNK = 64
    experiment.SCREEN_GROUP = 12
    experiment.POOL = (
        '杏队 苇队 槐队 楠队 叶队 崖队 沙江 沙队 池队 翠湾 潮队 岛队 梨队 榕队 莲队 桥队'.split(),
        '贺波 李琴 蒋洁 谭月 任华 严力 凌燕 夏林 魏昕 雷彤 陶文 丘露 毛霞 薛然 常玲 曹平 费宇 潘诚 蒙涛 钱楠 钟清 石英 卫晴 谢燕'.split(),
    )
    header = '{src}的名单：\n'
    entry = '{ent}分到的是{foil}。\n'
    intro = '同名草稿与正式名单会冲突。标记为草稿的记录尚未生效，答题只使用上面的正式名单。\n'
    experiment.NOTES = (
        intro+'草稿：\n'+header+entry,
        intro+'待审核的新名单（草稿）：\n'+header+entry*2,
        '优先级从高到低为正式名单、复核笔记、待审核草稿。\n待审核草稿：\n'+header+entry+'复核笔记（草稿截图）：\n'+header+entry,
        '下面是现场转录的名单抄件，和前面的正式名单不一致时，以正式名单为准。\n'+header+entry*2,
        '正式名单为答题依据。末尾附加名单只是排版演练。\n'+(header+entry)*3,
        '请使用正式名单里的值。后来收到的以下记录尚未替换正式名单。\n'+header+'{foil}分给了{ent}。\n给{ent}的动物是{foil}。\n{ent}：{foil}。\n',
        '下面的名单为旧版本。新版本是前面的正式名单。\n'+header+entry+'速查：{ent} → {foil}\n',
        '先前的正式名单有效。下列是同名旧名单，仅供核错。\n'+header+'给{ent}的动物是{foil}。\n错误抄件：{src} / {ent} / {foil}\n',
    )


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('design','screen','freeze','test','analyze'))
    mode=ap.parse_args().mode
    configure()
    io=experiment.io
    io.OUT=experiment.OUT
    io.setup()
    if mode=='design':
        experiment.materials()
        io.save('wrapper.json',dict(code=sha(__file__),design_sha256=sha(experiment.OUT/'design.json'),
                                   variants=experiment.NOTES,sources=experiment.POOL[0],entities=experiment.POOL[1],
                                   families=experiment.FAMILIES,seed=experiment.SEED))
        return
    wrapper=io.read('wrapper.json')
    assert wrapper['code']==sha(Path(__file__)) and wrapper['design_sha256']==sha(experiment.OUT/'design.json')
    if mode in ('screen','test'):
        experiment.run(mode)
    else:
        (analysis.freeze if mode=='freeze' else analysis.analyze)()


if __name__=='__main__':
    main()
