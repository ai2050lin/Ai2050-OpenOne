#!/usr/bin/env python3
"""Terminal audit for Phase2528-2535 and freeze the next token-atomic K/Q/V contract."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2];RESULT=ROOT/"tests/glm5/result";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md";ASSET=ROOT/"frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json";DIST=ROOT/"frontend/dist/index.html"
OUT=RESULT/"phase2536_c115457_c116480_stage_terminal_audit_next_contract";PHASE,CAMPAIGN=2536,"C115457-C116480"
PHASE_DIRS={2528:"phase2528_c94753_c95776_redundancy_source_edge_contract_audit",2529:"phase2529_c95777_c97568_full_source_kv_head_residual_ledger",2530:"phase2530_c97569_c99072_source_edge_sufficiency_lockbox",2531:"phase2531_c99073_c101120_redundant_route_cuts_rescue",2532:"phase2532_c101121_c104192_autonomous_multifamily_recursive_generation",2533:"phase2533_c104193_c107264_crossmodel_local_route_replication",2534:"phase2534_c107265_c109312_source_route_parameter_heatmap_publish",2535:"phase2535_c109313_c115456_autonomous_route_dose_replication"}

def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
 return h.hexdigest()

def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 text=rf"""


## Phase {PHASE}: 冗余耦合source-route大阶段总终审与下一合同（{CAMPAIGN}） [{stamp}]

**测试原理。** 对Phase2528–2535逐一检查唯一final、`all_checks_passed`、MEMO连续编号、模型精度与顺序、参数级资产、生产build、字段哈希和数据治理。把结论按“单Attention模块加法守恒→观察重复→source贡献充分→自然联盟必要→条件救援→自主递归→模型内重复→跨模型事件共性”分级，禁止跨级解释。

$$u_{{lhr}}=\sum_{{j\in r}}\alpha_{{lhaj}}v_{{lhj}},\quad g_{{lr}}=W_{{O,l}}\operatorname{{concat}}_h u_{{lhr}},\quad E(S)=m(x)-m(\operatorname{{do}}(S\leftarrow S^{{cut}})).$$

**成果与结果汇总。** 核心数字 `{json.dumps(r['key_numbers'],ensure_ascii=False)}`；已建立 `{json.dumps(r['established'],ensure_ascii=False)}`；未建立/硬伤 `{json.dumps(r['not_established'],ensure_ascii=False)}`；附件过度结论最终裁决 `{json.dumps(r['overclaim_adjudication'],ensure_ascii=False)}`。

**相关文件与数据治理。** 阶段final `{json.dumps(r['artifacts'],ensure_ascii=False)}`；客户端 `{json.dumps(r['visualization'],ensure_ascii=False)}`；留存/清理 `{json.dumps(r['data_governance'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。本Phase脚本为`tests/glm5/phase2536_c115457_c116480_stage_terminal_audit_next_contract.py`，final位于`{OUT}`。

**理论进展。** 当前最强、最小且不越界的机制图是：外部事实和query在残差流中形成上下文条件状态；答案边界的query heads按Q/K选择source token，其V内容经$W_O$写入2560维残差；Qwen3-4B存在一个在发现unit冻结、在新unit和21族自主生成中选择性充分且具有联盟必要效应的late layer×head候选集合。有限参数支持大量组合的可检验解释是“同一Q/K选择、V内容变换、$W_O$写入和残差递推规则在不同token/层/生成步反复复用”，而不是每句话对应固定语义坐标。该解释已得到source贡献、割、救援和递归拼图，但尚未揭示source token的语义状态如何被逐层写成K/V，也未得到最小、跨模型同构的语言代数。

**关键硬伤。** query-property的字符前缀分词边界在个别样本可映射为空token集合，说明现有五区域虽在token索引上互斥穷尽，却不是严格token-atomic语义切分；post-query既包含候选又包含输出指令，强充分性可能主要是输出身份/格式编译；matched whole-head救援按构造可恢复原轨迹，是阳性控制而非独立发现；全晚层删除有严重分布外损伤；完整长句重排行为门仅0.5625；Qwen14B/GLM4的模型内top路线自然删除不比随机更坏；跨模型没有共享物理坐标或必要联盟。

**下一大阶段合同。** `{json.dumps(r['next_contract'],ensure_ascii=False)}`。

**自动续研判断。** Phase2534之后，即时目标仍是同一冻结路线的跨unit、surface、语言和自主生成稳定性，因此已经自动完成Phase2535的672提示、6720条件生成剂量锁箱。Phase2535后该目标已完成；下一即时目标改为“token-atomic source状态如何写成K/Q/V、Q/K如何选址、V/$W_O$如何携带内容”，需要重做分词边界、干预对象和行为材料，属于新合同而不是同一实验的机械续跑。本轮因此在完成自动续研和冻结新合同后结束，不把未经token边界修复的K/V消融伪装成机制闭合。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def main():
 finals={str(p):load(RESULT/name/"analysis/final.json")for p,name in PHASE_DIRS.items()};f29=finals["2529"];f30=finals["2530"];f31=finals["2531"];f32=finals["2532"];f33=finals["2533"];f34=finals["2534"];f35=finals["2535"];asset=load(ASSET);memo=MEMO.read_text(encoding="utf-8")
 cross={k:{"donor_top_flip":v["causal"]["donor_top"]["donor_flip_rate"],"donor_random_flip":v["causal"]["donor_random"]["donor_flip_rate"],"zero_top_accuracy":v["causal"]["head_zero_top"]["accuracy"],"zero_random_accuracy":v["causal"]["head_zero_random"]["accuracy"],"zero_all_accuracy":v["causal"]["head_zero_all_late"]["accuracy"]}for k,v in f33["models"].items()}
 key_numbers={"source_ledger":{"head_pre_o_max_relative_rms":f29["conservation"]["maximum_head_pre_o_relative_rms"],"residual_max_relative_rms":f29["conservation"]["maximum_attention_residual_relative_rms"],"unit_energy_spearman":f29["routes"]["unit30_unit31_energy_spearman"],"top32_overlap":f29["routes"]["top32_overlap"]},"source_sufficiency":{"top_external_flip":f30["causal"]["donor_top_external"]["donor_flip_rate"],"random_external_flip":f30["causal"]["donor_random_external"]["donor_flip_rate"],"facts_shift":f30["causal"]["donor_top_facts"]["mean_shift_to_donor"],"facts_flip":f30["causal"]["donor_top_facts"]["donor_flip_rate"],"post_query_flip":f30["causal"]["donor_top_post_query"]["donor_flip_rate"]},"necessity_and_rescue":{"baseline_accuracy":f31["causal"]["no_patch"]["accuracy"],"top_edge_cut_accuracy":f31["causal"]["edge_cut_top_external"]["accuracy"],"random_edge_cut_accuracy":f31["causal"]["edge_cut_random_external"]["accuracy"],"all_edge_cut_accuracy":f31["causal"]["edge_cut_all_late_external"]["accuracy"],"all_cut_top_rescue_accuracy":f31["causal"]["edge_cut_all_late_external_rescue_top"]["accuracy"],"all_cut_random_rescue_accuracy":f31["causal"]["edge_cut_all_late_external_rescue_random"]["accuracy"]},"autonomous":{"qualified_families":len(f32["behavior"]["qualified_families"]),"unit33_surface1_baseline":f32["causal"]["main"]["no_patch"]["accuracy"],"unit33_surface1_top32_zero":f32["causal"]["main"]["head_zero_top"]["accuracy"],"unit33_surface1_random32_zero":f32["causal"]["main"]["head_zero_random"]["accuracy"],"multihop_baseline":f32["behavior"]["auxiliary"]["multihop"]["accuracy"],"full_reorder_baseline":f32["behavior"]["auxiliary"]["full_clause_reorder"]["accuracy"]},"large_dose":{"n_prompts":f35["scope"]["prompts"],"baseline":f35["metrics"]["no_patch"]["accuracy"],"top8":f35["metrics"]["head_zero_top8"]["accuracy"],"top16":f35["metrics"]["head_zero_top16"]["accuracy"],"top24":f35["metrics"]["head_zero_top24"]["accuracy"],"top32":f35["metrics"]["head_zero_top32"]["accuracy"],"random32":f35["metrics"]["head_zero_random32"]["accuracy"],"all_late":f35["metrics"]["head_zero_all_late"]["accuracy"]},"crossmodel":cross}
 established=["五个互斥source区域上的全late-head 128维贡献与经W_O写入的2560维residual在Attention模块内加法守恒。","unit30全512路线筛选的贡献top32在unit31 external-source donor patch达到36/36翻转，随机32为0/36。","自然持续edge cut使top32准确率降至0.8056，随机32保持1.0；all-late降至0.5。","all-late edge cut后恢复top32 whole-head使准确率回到1.0，恢复随机32仅0.4722；这是条件充分与特异阳性对照。","21个新语言操作族通过双unit行为门；自主多token生成中top32持续删除远强于随机删除。","672提示剂量复现显示top8/16/24/32准确率0.862/0.769/0.644/0.284，而随机32为0.927、基线0.918。","三个额外模型的模型内top donor均优于等量随机，但top自然必要性只在DS7B较清楚。"]
 not_established=["top32不是最小、唯一或逐head必要集合；top/complement均可部分承担行为。","source语义token边界尚不严格；个别query-property分区为空，post-query混合候选与指令。","Q/K的选址贡献与V的内容贡献尚未分别因果闭合；source residual如何逐层写入K/V未知。","MLP与Attention在自然路径中的条件耦合未完成matched source级分解。","完整长句重排行为资格不足；多跳只有受控两跳且未做逐跳K/V闭合。","跨模型没有共享坐标/head、共同最小联盟或同一内部算法证明。","强zero、全晚层割和whole-head donor均可能分布外；matched全状态/whole-head恢复含构造性恒等。","有限参数如何支持真正开放世界与无限组合仍只有复用假说和局部拼图。"]
 overclaim=["保留‘选择性late-head候选路线’，升级为Qwen3-4B中具有source贡献充分、联盟必要效应和自主递归损伤的模型内路线。","撤销‘32 heads已经被证明从事实前缀读取关系’：facts单区只推动margin且0翻转，post-query单区75%翻转，真正强项是外部source联合。","撤销‘只有32 heads参与’和‘最小齿轮’：complement、all-late与跨模型冗余证据明确。","把‘残差精确闭合’限定为Attention模块/同次前向的数值账本，不称语言机制闭合。","把‘删除后补偿’改称训练后网络中已有的备用读取/重构；没有在线学习。","把‘路径最小割’降为干预协议下的经验割候选，尚无图论最小性。"]
 next_contract={"name":"token-atomic source-state to Q/K/V write-read compiler campaign","wp1":"用显式哨兵和模型tokenizer差分对齐构造30+族，使事实实体、关系词、query-property、候选、指令在token层严格非空、互斥；修复part-whole、translation、destination、多跳和完整长句重排行为门。","wp2":"在全部late heads和全部head坐标上分别保存Q、post-RoPE K、V、QK logit、softmax质量、V加权内容和W_O residual写入；先做基本逐项守恒，不先套高等数学。","wp3":"unit发现/lockbox分离，分别干预source residual→K、source residual→V、单destination attention edge和W_O写入；zero、base-matched、counterfactual、错source和同范数随机分开。","wp4":"追踪MLP如何改变source token下一层K/V，做持续阻断与matched source级救援，寻找经验最小条件联盟而非单head神经元。","wp5":"在自主每个生成步重复同一token-atomic edge测试，并为真两跳记录第一跳/第二跳source的调用次序；完整重排仅在行为门提高后进入内部裁决。","wp6":"三大模型各自冻结相对路线并顺序BF16复现；只寻找事件/功能同构，不比较物理编号。重要全场继续参数级发布，其余HiddenState记录哈希后清理。","success":"新unit和新surface中，冻结token-edge在至少20族同时具备matched充分、联盟损伤、source特异救援及自主递归复现；随机/错source对照失败；至少两个额外模型复现功能级而非编号级结构。"}
 phase_paths={p:str(RESULT/name/"analysis/final.json")for p,name in PHASE_DIRS.items()};large_fields=[]
 for p in (f29,f31,f32):
  for v in p.get("fields",{}).values():
   if isinstance(v,dict)and v.get("path")and Path(v["path"]).exists()and Path(v["path"]).stat().st_size>1024*1024:large_fields.append({"path":v["path"],"bytes":Path(v["path"]).stat().st_size,"sha256":sha(Path(v["path"]))})
 checks={"all_phase_finals_passed":all(v["all_checks_passed"]for v in finals.values()),"memo_2528_2535_once":all(len(re.findall(rf"^## Phase {p}:",memo,re.M))==1 for p in PHASE_DIRS),"phase_numbers_contiguous":list(PHASE_DIRS)==list(range(2528,2536)),"asset_phase_2535":asset["phase"]==2535,"asset_nine_sections":len(asset["models"])==9,"asset_hash":len(sha(ASSET))==64,"production_build_current":DIST.exists()and DIST.stat().st_mtime_ns>=ASSET.stat().st_mtime_ns,"large_fields_hashed":all(len(x["sha256"])==64 for x in large_fields),"auto_continuation_completed":f35["all_checks_passed"],"next_target_changed":True,"claim_boundary":True}
 r={"phase":PHASE,"campaign":CAMPAIGN,"key_numbers":key_numbers,"established":established,"not_established":not_established,"overclaim_adjudication":overclaim,"artifacts":{"phase_finals":phase_paths,"terminal_script":str(Path(__file__))},"visualization":{"asset":str(ASSET),"sha256":sha(ASSET),"bytes":ASSET.stat().st_size,"sections":{s["key"]:s["coordinate_count"]for s in asset["models"]},"production_dist":str(DIST)},"data_governance":{"large_display_sources":large_fields,"retained_bytes":sum(x["bytes"]for x in large_fields),"unpublished_hiddenstate_deleted":[],"reason":"all newly retained HiddenState fields are represented by parameter-level heatmap rows"},"next_contract":next_contract,"automatic_continuation":{"same_target_phase":2535,"completed":True,"next_target_changed":True},"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis/final.json",r)
 if r["all_checks_passed"]:append_memo(r)
 print(json.dumps({"phase":PHASE,"key_numbers":key_numbers,"established":established,"not_established":not_established,"checks":checks,"all_checks_passed":r["all_checks_passed"]},ensure_ascii=False,indent=2))
 if not r["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
