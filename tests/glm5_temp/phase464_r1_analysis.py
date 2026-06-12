"""Phase 464 R1结果分析"""
import json, sys, numpy as np

def load_results(model, r=1):
    path = f"results/glm5/phase464_{model}_r{r}.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

print("="*80)
print("Phase 464 R1 三模型关键结果对比")
print("="*80)

for model in ["qwen3", "deepseek7b", "glm4"]:
    try:
        d = load_results(model)
        print(f"\n{'='*60}")
        print(f"模型: {model} ({d.get('model_class','')}, {d.get('n_layers','')}L)")
        print(f"{'='*60}")
        
        # Exp1: 正交分解修复
        if "exp1_orthogonal_fix" in d:
            exp1 = d["exp1_orthogonal_fix"]
            print("\n--- Exp1: 正交分解(修复后) ---")
            for layer_key in sorted(exp1.keys()):
                items = exp1[layer_key]
                if not items:
                    continue
                first_item = list(items.values())[0] if items else {}
                cos_val = first_item.get("cos_sem_lang", "N/A")
                new_ratio = first_item.get("sem_only_ratio_fixed", "N/A")
                old_ratio = first_item.get("sem_only_ratio_old", "N/A")
                theory = first_item.get("theoretical_ratio", "N/A")
                err = first_item.get("ratio_error", "N/A")
                print(f"  {layer_key}: cos(sem,lang)={cos_val:.4f if isinstance(cos_val, float) else cos_val}, "
                      f"NEW_ratio={new_ratio:.4f if isinstance(new_ratio, float) else new_ratio}, "
                      f"OLD_ratio={old_ratio:.4f if isinstance(old_ratio, float) else old_ratio}, "
                      f"theory={theory:.4f if isinstance(theory, float) else theory}, "
                      f"err={err:.4f if isinstance(err, float) else err}")
        
        # Exp3: 跨类别holdout
        if "exp3_cross_category_holdout" in d:
            exp3 = d["exp3_cross_category_holdout"]
            print("\n--- Exp3: 跨类别holdout ---")
            for key in sorted(exp3.keys()):
                val = exp3[key]
                cat = val.get("category", "?")
                beta = val.get("beta", "?")
                avg_sel = val.get("avg_selectivity", 0)
                print(f"  {key}: cat={cat} beta={beta} avg_sel={avg_sel:.3f}")
        
        # Exp4: 语言轴干预
        if "exp4_language_axis_intervention" in d:
            exp4 = d["exp4_language_axis_intervention"]
            print("\n--- Exp4: 语言轴因果干预 ---")
            for key in sorted(exp4.keys()):
                val = exp4[key]
                li = val.get("patch_layer", "?")
                beta = val.get("beta", "?")
                eff_rank = val.get("eff_rank", "?")
                pos_en = val.get("pos_en_delta", 0)
                neg_en = val.get("neg_en_delta", 0)
                pos_zh = val.get("pos_zh_delta", 0)
                neg_zh = val.get("neg_zh_delta", 0)
                print(f"  {key}: eff_rank={eff_rank:.3f if isinstance(eff_rank, float) else eff_rank} "
                      f"beta={beta} +enΔ={pos_en:.2f} -enΔ={neg_en:.2f} +zhΔ={pos_zh:.2f} -zhΔ={neg_zh:.2f}")
        
        # Exp5: 翻译维度
        if "exp5_translate_dimension" in d:
            exp5 = d["exp5_translate_dimension"]
            print("\n--- Exp5: 翻译控制维度 ---")
            for key in sorted(exp5.keys()):
                val = exp5[key]
                eff_rank = val.get("eff_rank", "?")
                top1 = val.get("top1_energy_ratio", "?")
                print(f"  {key}: eff_rank={eff_rank:.3f if isinstance(eff_rank, float) else eff_rank}, "
                      f"top1={top1:.3f if isinstance(top1, float) else top1}")
        
        # Exp6: 策略指标
        if "exp6_strategy_indicators" in d:
            exp6 = d["exp6_strategy_indicators"]
            print("\n--- Exp6: 策略指标 ---")
            for k, v in exp6.items():
                if v is not None:
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
    except Exception as e:
        print(f"\n{model}: Error - {e}")

print("\n\n" + "="*80)
print("三模型Exp5翻译控制维度对比")
print("="*80)
print(f"{'Layer':<10} {'Qwen3':<10} {'DS7B':<10} {'GLM4':<10}")
for layer_label in ["L6", "L12", "L9", "L13", "L18", "L20", "L24", "L26", "L33", "L25", "L37"]:
    vals = []
    for model in ["qwen3", "deepseek7b", "glm4"]:
        try:
            d = load_results(model)
            if "exp5_translate_dimension" in d:
                exp5 = d["exp5_translate_dimension"]
                if layer_label in exp5:
                    vals.append(f"{exp5[layer_label]['eff_rank']:.3f}")
                else:
                    vals.append("-")
            else:
                vals.append("-")
        except:
            vals.append("-")
    if any(v != "-" for v in vals):
        print(f"{layer_label:<10} {vals[0]:<10} {vals[1]:<10} {vals[2]:<10}")
