"""Pre-model grammar correction, preserving the original material revision."""
import shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2641_matched_operation_contract import *
path=OUT/'material/cases.json';backup=OUT/'material/cases.pre_article_fix.json'
if backup.exists():raise RuntimeError('preflight already applied')
assert not (RESULT/'phase2642_matched_operation_behavior/behavior/greedy.jsonl').exists()
old=read(path);before=sha(path)
tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);new=build(tok)
changed=[a['case_id'] for a,b in zip(old,new) if a['prompt']!=b['prompt']]
assert all(a['case_id']==b['case_id'] and a['target']==b['target'] for a,b in zip(old,new))
shutil.copy2(path,backup);save(path,new)
save(OUT/'protocol/material_revision.json',{'reason':'before any model test: English a apple -> an apple in taxonomy only','changed_prompts':len(changed),'changed_case_ids':changed,
    'before_sha256':before,'after_sha256':sha(path),'before_model_run':True})
stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
with MEMO.open('a',encoding='utf-8') as f:f.write(f'\n\n**Phase2641 模型运行前语料修订 [{stamp}]。** 英语分类句不定冠词按词首元音修正，{len(changed)}条a apple改为an apple；实体、目标和析因条件不变，尚未执行任何本批模型测试。原语料版本另存cases.pre_article_fix.json，前后哈希及精确case清单在protocol/material_revision.json。\n')
print('pre-model article correction',len(changed),flush=True)
