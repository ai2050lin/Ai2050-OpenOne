# -*- coding: utf-8 -*-
"""ENT-XF 标注校准：被测模型对颜色 / 大小提示的自由回答（不给选项）。

    python scripts/calibrate_ent_freeresp.py --model models/hf/qwen3-4b \
        --out research/family_runs/entangle/ENT-XF-v2-freeresponse-qwen3-4b.json

只用于复核词表标注、扩充可接受答案集（每个名词可有多个可接受颜色）；不得据此挑选或剔除名词——
否则行为门通过率会被按构造抬高（Phase 250）。不是正式运行，不进入合并统计。

每个名词 × 语言 × 属性：贪心续写 1 次 + 温度采样 n 次（固定种子），最多 max_new 个 token；
从续写中抽取第一个颜色词 / 大小词（词表见 COLOR_PATTERNS / SIZE_PATTERNS），未识别的原样保存。
提示与 ENT-XF v2 的行为 probe 完全相同（scripts/build_ent_grammar.py 的 PROBES）。
"""
import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 中文：较长的形式在前（金黄 先于 黄，粉红 先于 红）
COLOR_PATTERNS = {
    'zh': [('金黄', 'yellow'), ('橙黄', 'orange'), ('橘黄', 'orange'), ('橘红', 'orange'), ('粉红', 'pink'),
           ('紫红', 'purple'), ('棕黄', 'brown'), ('黄褐', 'brown'), ('灰褐', 'brown'), ('黑褐', 'brown'),
           ('咖啡', 'brown'), ('透明', 'transparent'), ('银', 'gray'), ('金', 'yellow'), ('青', 'green'),
           ('红', 'red'), ('黄', 'yellow'), ('绿', 'green'), ('蓝', 'blue'), ('白', 'white'), ('黑', 'black'),
           ('棕', 'brown'), ('褐', 'brown'), ('橙', 'orange'), ('橘', 'orange'), ('紫', 'purple'), ('粉', 'pink'),
           ('灰', 'gray')],
    'en': [('golden', 'yellow'), ('gold', 'yellow'), ('yellow', 'yellow'), ('red', 'red'), ('green', 'green'),
           ('blue', 'blue'), ('white', 'white'), ('black', 'black'), ('brown', 'brown'), ('tan', 'brown'),
           ('orange', 'orange'), ('purple', 'purple'), ('violet', 'purple'), ('pink', 'pink'), ('gray', 'gray'),
           ('grey', 'gray'), ('silver', 'gray'), ('transparent', 'transparent'), ('translucent', 'transparent'),
           ('clear', 'transparent')],
}
SIZE_PATTERNS = {
    'zh': [('大', 'L'), ('高', 'L'), ('长', 'L'), ('小', 'S'), ('矮', 'S'), ('短', 'S')],
    'en': [('bigger', 'L'), ('larger', 'L'), ('taller', 'L'), ('heavier', 'L'), ('longer', 'L'),
           ('smaller', 'S'), ('shorter', 'S'), ('lighter', 'S'), ('tinier', 'S')],
}


def first_label(text, lang, patterns):
    """续写中第一个出现的标签词（按出现位置；同一位置取较长的形式）。"""
    best = None
    for w, lab in patterns[lang]:
        m = re.search(r'\b' + w + r'\b' if lang == 'en' else re.escape(w), text.lower() if lang == 'en' else text)
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(w) > best[2])):
            best = (m.start(), lab, len(w))
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--max-new', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', default=None, help='只跑 lang/task，如 zh/color（诊断候选提示用）')
    ap.add_argument('--prompt', default=None, help='覆盖该 lang/task 的提示模板（{X} / {E}），只用于诊断候选提示')
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec_b = importlib.util.spec_from_file_location('build_ent_grammar', ROOT / 'scripts' / 'build_ent_grammar.py')
    B = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(B)
    spec_r = importlib.util.spec_from_file_location('runner_v3', ROOT / 'shared' / 'dist_protocol' / 'runner_v3.py')
    R = importlib.util.module_from_spec(spec_r)
    spec_r.loader.exec_module(R)

    rows = B.read_lexicon()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    t0 = time.time()
    out = []
    for task, patterns in (('color', COLOR_PATTERNS), ('size', SIZE_PATTERNS)):
        for lang in ('zh', 'en'):
            if args.only and args.only != f'{lang}/{task}':
                continue
            tmpl = args.prompt if args.prompt else B.PROBES[(lang, task)]['prompt']
            for r in rows:
                prompt = tmpl.replace('{X}', r['zh']).replace('{E}', r['en'])
                ids = tok(prompt, return_tensors='pt').to('cuda')
                n = ids['input_ids'].shape[1]
                with torch.inference_mode():
                    g = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    torch.manual_seed(args.seed)
                    s = model.generate(**ids, max_new_tokens=args.max_new, do_sample=True,
                                       temperature=args.temperature, top_p=1.0, top_k=0,
                                       num_return_sequences=args.samples, pad_token_id=tok.eos_token_id)
                greedy = tok.decode(g[0, n:], skip_special_tokens=True)
                samples = [tok.decode(x[n:], skip_special_tokens=True) for x in s]
                out.append({'id': r['id'], 'zh': r['zh'], 'en': r['en'], 'lang': lang, 'task': task,
                            'prompt': prompt, 'greedy': greedy, 'greedy_label': first_label(greedy, lang, patterns),
                            'samples': samples,
                            'sample_labels': dict(Counter(str(first_label(x, lang, patterns)) for x in samples))})
    res = {'schema': 'ent_freeresp.v1', 'kind': 'calibration',
           'note': '标注校准用的自由回答；只用于复核标注与扩充可接受答案集，不用于挑选名词；不是正式运行结果',
           'model_snapshot': R.model_snapshot(args.model), 'torch': torch.__version__,
           'transformers': transformers.__version__, 'dtype': 'bfloat16',
           'decode': {'greedy': True, 'samples': args.samples, 'temperature': args.temperature, 'top_p': 1.0,
                      'top_k': 0, 'max_new_tokens': args.max_new, 'seed': args.seed},
           'only': args.only, 'prompt_override': args.prompt,
           'lexicon': str(B.LEXICON.relative_to(ROOT)), 'seconds': round(time.time() - t0, 1), 'items': out}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', args.out, len(out), 'prompts', res['seconds'], 's')


if __name__ == '__main__':
    main()
