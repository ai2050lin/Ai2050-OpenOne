"""Second, separately frozen cohort with graded misleading demonstrations."""
import source_binding_qkv_low_margin as experiment
import source_binding_qkv_low_margin_analysis as analysis


def configure():
    experiment.OUT = experiment.OUT.parent / 'low-margin-expanded'
    experiment.SEED += 1
    experiment.FAMILIES = 144
    experiment.CHUNK = 64
    experiment.SCREEN_GROUP = 12
    experiment.POOL = (
        '雪队 枫队 桃队 荷队 梅队 霜队 虹队 月队 雁队 鹏队 橙队 柏队 峰队 湖队 舟队 翠队'.split(),
        '秦川 刘畅 程锦 周航 程玲 董莉 李舟 孔琳 罗雨 蒋云 沈杨 徐芳 苏静 韩宇 周悦 胡乐 孟婷 史晴 宋杨 顾涛 高原 梁琴 陆飞 秦明'.split(),
    )
    # Preserve the reference list and the scored question, and vary the context
    # before the question. A repeat is always labelled as a draft, example or
    # wrong record. The grading rule is stated before the distractor.
    exact = '上面的正式名单优先于示例。下列问答只作格式示例，答案与正式名单无关。\n'
    copied = '{question}\n参考答案：{foil}\n'
    same = '只使用上面正式名单。下面按时间从新到旧展示作废草稿，所有草稿都不参与核对。\n'
    old = '{src}的名单：{ent}分到的是{foil}。\n'
    plain = '以下词语都是无关的示例词，作答时查正式名单。\n'
    experiment.NOTES = (
        exact+copied,
        exact+copied*2,
        exact+copied*3,
        exact+copied.replace('参考答案：','Answer:\n')*2,
        exact+copied.replace('参考答案：','Answer:\n')*3,
        same+old*2,
        same+old*4,
        same+old*6,
        plain+'示例词：'+'{foil}、'*8+'{foil}。\n',
        '请根据正式名单作答。以下记录只是展示一个容易混淆的例子。\n'+copied.replace('参考答案：','答：')*3,
    )


def main():
    import argparse
    from pathlib import Path
    from source_binding_design import sha

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
                                   variants=experiment.NOTES,families=experiment.FAMILIES,seed=experiment.SEED))
        return
    wrapper=io.read('wrapper.json')
    assert wrapper['code']==sha(Path(__file__)) and wrapper['design_sha256']==sha(experiment.OUT/'design.json')
    if mode in ('screen','test'):
        experiment.run(mode)
    else:
        (analysis.freeze if mode=='freeze' else analysis.analyze)()


if __name__=='__main__':
    main()
