"""
Phase 67: 信号压缩理论 (SCT) 验证
============================================

Phase 66的关键发现:
1. MPS"乘性耦合"被否定: joint/sum=0.51-1.26，不是>>1
2. MLP gating翻转只占direction_add效果的18% → gating不是核心
3. 中层信号缩放严重亚线性: β增加45x→effect只增加6.4x
4. L0有phase transition(β=1.0)，深层没有
5. subspace_swap比direction_add强6-57x → 绕过了压缩

核心假设 (SCT):
  信号在Transformer中层被严重压缩
  压缩来源: LayerNorm + Residual Dilution
  direction_add失败 = 信号被压缩，不是gating或乘性路径

Phase 67的三个核心实验:
A. 信号传播追踪 — 注入固定δ，追踪每层的信号幅度
B. LN压缩实验 — 对比有LN和无LN的信号传播
C. L0入口层控制 — 利用L0的phase transition做可靠控制
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from model_utils import load_model, get_model_info, release_model, get_layers, get_layer_weights

REGULAR_NVA = [
    ("cat","cats","runs","run"),("dog","dogs","walks","walk"),("bird","birds","flies","fly"),
    ("girl","girls","reads","read"),("boy","boys","sings","sing"),("horse","horses","jumps","jump"),
    ("bear","bears","sleeps","sleep"),("snake","snakes","crawls","crawl"),
    ("frog","frogs","swims","swim"),("fox","foxes","hunts","hunt"),("king","kings","rules","rule"),
    ("student","students","studies","study"),("teacher","teachers","speaks","speak"),
    ("doctor","doctors","helps","help"),("tree","trees","grows","grow"),("car","cars","moves","move"),
    ("queen","queens","leads","lead"),("driver","drivers","drives","drive"),
    ("worker","workers","builds","build"),("player","players","wins","win"),
    ("writer","writers","writes","write"),("farmer","farmers","plants","plant"),
    ("nurse","nurses","cares","care"),("soldier","soldiers","fights","fight"),
    ("rabbit","rabbits","hops","hop"),("whale","whales","dives","dive"),
    ("eagle","eagles","soars","soar"),("lion","lions","roars","roar"),
    ("tiger","tigers","hunts","hunt"),("wolf","wolves","howls","howl"),
    ("child","children","plays","play"),("mouse","mice","squeaks","squeak"),
    ("goose","geese","swims","swim"),("man","men","works","work"),
    ("woman","women","sings","sing"),("tooth","teeth","aches","ache"),
    ("foot","feet","steps","step"),("person","people","thinks","think"),
]


def make_sentences(nva):
    sing, plur, sv, pv = nva
    return (f"The {sing} {sv}", f"The {plur} {pv}", sv, pv)


# ============================================================
# 实验 A: 信号传播追踪
# ============================================================
def experiment_a_signal_propagation(model, tokenizer, device, model_info, n_test=40):
    """
    核心问题: 扰动信号δ在Transformer各层如何衰减?
    
    方法:
    1. 在layer i注入固定方向δ(单位向量)
    2. 用hook捕获layer i+1, i+2, ..., L-1的hidden states
    3. 计算: δ_residual(l) = h_intervened(l) - h_baseline(l)
    4. 追踪: ||δ_residual(l)|| 的变化曲线
    
    SCT预测:
    - 中层衰减最快
    - L0衰减较慢(skip connection传播)
    - L27衰减最慢(直接到输出)
    """
    print("\n" + "="*70)
    print("实验 A: 信号传播追踪")
    print("="*70)
    
    layers = get_layers(model)
    n_layers = model_info.n_layers
    
    # 注入层: 均匀采样
    inject_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    inject_layers = sorted(set(l for l in inject_layers if l < n_layers))
    
    beta = 2.0  # 注入强度
    
    # 收集: {inject_layer: {downstream_layer: [amplitude_ratios]}}
    results = {il: {} for il in inject_layers}
    
    for si in range(min(n_test, len(REGULAR_NVA))):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        
        for inject_li in inject_layers:
            # Step 1: 获取baseline hidden states
            captured_baseline = {}
            def make_capture_hook(key, store):
                def hook(module, input, output):
                    out = output[0].detach().clone().float()
                    store[key] = out
                return hook
            
            hooks_b = []
            for dl in range(n_layers):
                h = layers[dl].register_forward_hook(make_capture_hook(f'L{dl}', captured_baseline))
                hooks_b.append(h)
            
            with torch.no_grad():
                _ = model(**sing_ids)
            
            for h in hooks_b:
                h.remove()
            
            # Step 2: 注入扰动 + 捕获下游hidden states
            # 使用随机方向(避免direction本身的偏差)
            if inject_li == 0:
                # 在embedding层注入
                # 使用plur的hidden state作为"方向"
                continue  # 跳过，用标准方法
            else:
                # 在inject_li层注入随机方向
                if f'L{inject_li}' not in captured_baseline:
                    continue
                
                h_at_inject = captured_baseline[f'L{inject_li}']
                d_model = h_at_inject.shape[-1]
                random_dir = torch.randn(d_model, device=device, dtype=torch.float16)
                random_dir = random_dir / random_dir.norm()
            
            captured_intervened = {}
            hooks_i = []
            for dl in range(n_layers):
                h = layers[dl].register_forward_hook(make_capture_hook(f'L{dl}', captured_intervened))
                hooks_i.append(h)
            
            # 注入hook
            def inject_hook(module, input, output):
                out = output[0].detach().clone()
                pos = min(s_vp, out.shape[1]-1)
                out[0, pos, :] += (beta * random_dir).to(out.dtype)
                return (out,) + output[1:] if isinstance(output, tuple) else out
            
            h_inject = layers[inject_li].register_forward_hook(inject_hook)
            
            with torch.no_grad():
                _ = model(**sing_ids)
            
            h_inject.remove()
            for h in hooks_i:
                h.remove()
            
            # Step 3: 计算信号传播
            for dl in range(inject_li + 1, n_layers):
                if f'L{dl}' in captured_baseline and f'L{dl}' in captured_intervened:
                    h_b = captured_baseline[f'L{dl}'][0, min(s_vp, captured_baseline[f'L{dl}'].shape[1]-1), :].cpu().float().numpy()
                    h_i = captured_intervened[f'L{dl}'][0, min(s_vp, captured_intervened[f'L{dl}'].shape[1]-1), :].cpu().float().numpy()
                    delta = h_i - h_b
                    amp = np.linalg.norm(delta)
                    
                    if inject_li not in results:
                        results[inject_li] = {}
                    if dl not in results[inject_li]:
                        results[inject_li][dl] = []
                    results[inject_li][dl].append(amp)
            
            # 清理
            del captured_baseline, captured_intervened
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 汇总
    print("\n--- 实验 A 结果: 信号传播追踪 ---")
    print(f"注入强度β={beta}, 随机方向")
    
    # 对每个注入层，画出信号衰减曲线
    for inject_li in inject_layers:
        if not results[inject_li]:
            continue
        
        print(f"\n从L{inject_li}注入, 信号幅度衰减:")
        downstream_layers = sorted(results[inject_li].keys())
        
        # 归一化: 以第一个下游层为基准
        if downstream_layers:
            first_amp = np.mean(results[inject_li][downstream_layers[0]]) if results[inject_li][downstream_layers[0]] else 1.0
            print(f"  {'Layer':<8} {'|δ|':<10} {'ratio':<10}")
            for dl in downstream_layers[:8]:  # 限制输出
                amp = np.mean(results[inject_li][dl]) if results[inject_li][dl] else 0
                ratio = amp / first_amp if first_amp > 1e-8 else 0
                print(f"  L{dl:<7} {amp:<10.4f} {ratio:<10.3f}")
            if len(downstream_layers) > 8:
                print(f"  ... ({len(downstream_layers)-8} more layers)")
    
    # ★ 关键分析: 衰减率 vs 层位置
    print("\n[★ 关键分析: 1层衰减率 (每经过1层信号保留比例)]")
    layer_decay = defaultdict(list)
    
    for inject_li in inject_layers:
        downstream_layers = sorted(results[inject_li].keys())
        for i in range(len(downstream_layers) - 1):
            dl1 = downstream_layers[i]
            dl2 = downstream_layers[i + 1]
            if results[inject_li][dl1] and results[inject_li][dl2]:
                a1 = np.mean(results[inject_li][dl1])
                a2 = np.mean(results[inject_li][dl2])
                if a1 > 1e-8:
                    decay = a2 / a1
                    # 标记下游层位置
                    mid_layer = (dl1 + dl2) // 2
                    layer_decay[mid_layer].append(decay)
    
    if layer_decay:
        print(f"  {'Layer range':<15} {'Avg retention':<15}")
        for l in sorted(layer_decay.keys()):
            avg_ret = np.mean(layer_decay[l])
            print(f"  L{l:<13} {avg_ret:<15.3f}")
    
    return results


# ============================================================
# 实验 B: LN压缩实验
# ============================================================
def experiment_b_ln_compression(model, tokenizer, device, model_info, n_test=50):
    """
    核心问题: LayerNorm对信号压缩的贡献有多大?
    
    方法:
    1. 标准direction_add (有LN)
    2. 在注入点后，修改LN的输出使其不压缩信号
    
    更精确的方法:
    比较direction_add前后的hidden state:
    - 注入前: h_before_LN → LN(h_before_LN) 
    - 注入后: h_before_LN + δ → LN(h_before_LN + δ)
    - LN压缩比: ||LN(h+δ) - LN(h)|| / ||δ||
    
    SCT预测: LN压缩比 < 1 (信号被压缩)
    """
    print("\n" + "="*70)
    print("实验 B: LN压缩量化")
    print("="*70)
    
    layers = get_layers(model)
    target_layers = [0, 5, 10, 15, model_info.n_layers - 1]
    target_layers = [l for l in target_layers if l < model_info.n_layers]
    
    beta = 2.0
    
    # 收集: {layer: [raw_ratio, ln_ratio, ln_compression_factor]}
    results = {l: {
        'raw_delta_norm': [],
        'ln_delta_norm': [],
        'ln_compression': [],  # ||LN(h+δ)-LN(h)|| / ||δ||
    } for l in target_layers}
    
    for si in range(min(n_test, len(REGULAR_NVA))):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        
        # 获取plur的direction
        plur_ids = tokenizer(f"The {REGULAR_NVA[si][1]} {REGULAR_NVA[si][3]}", return_tensors="pt").to(device)
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        with torch.no_grad():
            s_out = model(**sing_ids, output_hidden_states=True)
            p_out = model(**plur_ids, output_hidden_states=True)
        
        for li in target_layers:
            if li+1 >= len(s_out.hidden_states) or li+1 >= len(p_out.hidden_states):
                continue
            
            s_h = s_out.hidden_states[li+1][0, min(s_vp, s_out.hidden_states[li+1].shape[1]-1), :]
            p_h = p_out.hidden_states[li+1][0, min(p_vp, p_out.hidden_states[li+1].shape[1]-1), :]
            
            direction = (p_h - s_h).float()
            d_norm = direction.norm().item()
            if d_norm < 1e-8:
                continue
            direction = direction / d_norm
            
            # 计算LN压缩
            # 获取该层的LN前的hidden state
            # 方法: 用hook捕获LN输入
            ln_input_data = {}
            
            # 找到该层的LN
            layer = layers[li]
            ln_module = None
            for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
                if hasattr(layer, ln_name):
                    ln_module = getattr(layer, ln_name)
                    break
            
            if ln_module is None:
                continue
            
            def ln_capture_hook(module, input, output):
                ln_input_data['input'] = input[0].detach().clone().float()
                ln_input_data['output'] = output[0].detach().clone().float() if not isinstance(output, tuple) else output[0].detach().clone().float()
            
            h = ln_module.register_forward_hook(ln_capture_hook)
            
            # Baseline
            with torch.no_grad():
                _ = model(**sing_ids)
            h.remove()
            
            if 'input' not in ln_input_data or 'output' not in ln_input_data:
                continue
            
            h_before_ln = ln_input_data['input']  # [1, seq, d_model]
            pos = min(s_vp, h_before_ln.shape[1]-1)
            h_vec = h_before_ln[0, pos, :]  # [d_model]
            
            # 加扰动的版本
            h_plus_delta = h_vec.cpu() + beta * direction.cpu()
            
            # 手动计算LN
            def manual_ln(h, weight, eps=1e-5):
                mean = h.mean()
                std = h.std()
                return weight * (h - mean) / (std + eps)
            
            # 获取LN weight
            ln_weight = ln_module.weight.detach().cpu().float()
            
            h_ln = manual_ln(h_vec.cpu(), ln_weight)
            h_plus_ln = manual_ln(h_plus_delta, ln_weight)
            
            # 原始扰动幅度
            raw_delta_norm = np.linalg.norm((beta * direction.cpu()).numpy())
            
            # LN后扰动幅度
            ln_delta = (h_plus_ln - h_ln).numpy()
            ln_delta_norm = np.linalg.norm(ln_delta)
            
            # LN压缩比
            compression = ln_delta_norm / raw_delta_norm if raw_delta_norm > 1e-8 else 0
            
            results[li]['raw_delta_norm'].append(raw_delta_norm)
            results[li]['ln_delta_norm'].append(ln_delta_norm)
            results[li]['ln_compression'].append(compression)
        
        del s_out, p_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print("\n--- 实验 B 结果: LN压缩量化 ---")
    print(f"{'Layer':<8} {'raw|δ|':<12} {'LN|δ|':<12} {'compression':<14} {'信号保留%':<12}")
    for li in target_layers:
        raw = results[li]['raw_delta_norm']
        ln = results[li]['ln_delta_norm']
        comp = results[li]['ln_compression']
        if comp:
            print(f"L{li:<7} {np.mean(raw):<12.4f} {np.mean(ln):<12.4f} {np.mean(comp):<14.4f} {np.mean(comp)*100:<12.1f}%")
    
    # ★ 关键判定
    print("\n[★ 关键判定: LN是否是主要压缩源?]")
    avg_compression = np.mean([np.mean(results[li]['ln_compression']) 
                              for li in target_layers if results[li]['ln_compression']])
    if avg_compression < 0.5:
        print(f"  平均LN压缩比={avg_compression:.3f} → LN是主要压缩源 ★")
    elif avg_compression < 0.8:
        print(f"  平均LN压缩比={avg_compression:.3f} → LN是部分压缩源")
    else:
        print(f"  平均LN压缩比={avg_compression:.3f} → LN不是主要压缩源")
    
    return results


# ============================================================
# 实验 C: L0入口层直接控制
# ============================================================
def experiment_c_l0_control(model, tokenizer, device, model_info, n_test=80):
    """
    Phase 66发现: L0有phase transition(β=1.0)，在β=8时效果跃升
    
    核心问题: 在L0做大β direction_add是否可以可靠控制输出?
    
    验证:
    1. β=8 (超过phase transition阈值)的效果稳定性
    2. 成功率: sing→plur翻转的比例
    3. 对比L0 vs L10 vs L27的控制效率
    """
    print("\n" + "="*70)
    print("实验 C: L0入口层直接控制")
    print("="*70)
    
    layers = get_layers(model)
    target_layers = [0, 10, model_info.n_layers - 1]
    target_layers = [l for l in target_layers if l < model_info.n_layers]
    
    betas = [2.0, 8.0, 16.0, 32.0]
    
    # 收集: {layer: {beta: {'flip_rate': [], 'effect': []}}}
    results = {l: {b: {'flip_rate': [], 'effect': []} for b in betas} for l in target_layers}
    
    for si in range(min(n_test, len(REGULAR_NVA))):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            continue
        
        with torch.no_grad():
            sing_out = model(**sing_ids, output_hidden_states=True)
            plur_out = model(**plur_ids, output_hidden_states=True)
        
        # baseline: sing应该偏好singular verb
        baseline_logits = sing_out.logits[0, -1, :]
        baseline_s = baseline_logits[sv_tok[0]].item()
        baseline_p = baseline_logits[pv_tok[0]].item()
        baseline_diff = baseline_s - baseline_p  # positive = correct
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        for li in target_layers:
            if li+1 >= len(sing_out.hidden_states) or li+1 >= len(plur_out.hidden_states):
                continue
            
            s_h = sing_out.hidden_states[li+1][0, min(s_vp, sing_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = plur_out.hidden_states[li+1][0, min(p_vp, plur_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            
            direction = p_h - s_h
            d_norm = np.linalg.norm(direction)
            if d_norm < 1e-8:
                continue
            direction = direction / d_norm
            direction_t = torch.tensor(direction, dtype=torch.float16, device=device)
            
            for beta in betas:
                def inject_hook(module, input, output):
                    out = output[0].detach().clone()
                    pos = min(s_vp, out.shape[1]-1)
                    out[0, pos, :] += (beta * direction_t).to(out.dtype)
                    return (out,) + output[1:] if isinstance(output, tuple) else out
                
                h = layers[li].register_forward_hook(inject_hook)
                with torch.no_grad():
                    interv_out = model(**sing_ids)
                h.remove()
                
                interv_logits = interv_out.logits[0, -1, :]
                interv_s = interv_logits[sv_tok[0]].item()
                interv_p = interv_logits[pv_tok[0]].item()
                interv_diff = interv_s - interv_p
                
                effect = interv_diff - baseline_diff  # negative = 成功翻转向plur
                flip = 1 if interv_diff < 0 else 0  # 翻转成功
                
                results[li][beta]['effect'].append(effect)
                results[li][beta]['flip_rate'].append(flip)
                
                del interv_out
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print("\n--- 实验 C 结果: L0入口层直接控制 ---")
    print(f"{'Layer':<8} {'β':<6} {'mean_effect':<14} {'flip_rate%':<12} {'方向正确%':<12}")
    
    for li in target_layers:
        for beta in betas:
            effs = results[li][beta]['effect']
            flips = results[li][beta]['flip_rate']
            if effs:
                mean_eff = np.mean(effs)
                flip_pct = np.mean(flips) * 100
                # 方向正确 = effect < 0 (push toward plural)
                correct = np.mean([1 if e < 0 else 0 for e in effs]) * 100
                print(f"L{li:<7} {beta:<6.1f} {mean_eff:<14.4f} {flip_pct:<12.1f} {correct:<12.1f}")
    
    # ★ 关键判定
    print("\n[★ 关键判定: L0是否是最有效的控制层?]")
    
    # 在β=8时比较各层
    target_beta = 8.0
    for li in target_layers:
        effs = results[li][target_beta]['effect']
        flips = results[li][target_beta]['flip_rate']
        if effs:
            mean_eff = np.mean(effs)
            correct = np.mean([1 if e < 0 else 0 for e in effs]) * 100
            flip_pct = np.mean(flips) * 100
            print(f"  L{li} (β={target_beta}): 方向正确率={correct:.1f}%, 完全翻转率={flip_pct:.1f}%, 平均effect={mean_eff:.4f}")
    
    # 成功率随β的变化
    print("\n[★ 翻转成功率随β的变化]")
    for li in target_layers:
        rates = []
        for beta in betas:
            flips = results[li][beta]['flip_rate']
            if flips:
                rates.append(f"β={beta}:{np.mean(flips)*100:.0f}%")
        if rates:
            print(f"  L{li}: {', '.join(rates)}")
    
    return results


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 67: SCT验证")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--exp", type=str, default="all",
                       choices=["a", "b", "c", "all"])
    args = parser.parse_args()
    
    print("="*70)
    print(f"Phase 67: 信号压缩理论 (SCT) 验证")
    print(f"模型: {args.model}, n_test={args.n_test}")
    print("="*70)
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"模型信息: {model_info.n_layers}层, d_model={model_info.d_model}")
    
    results = {}
    t0 = time.time()
    
    try:
        if args.exp in ["b", "all"]:
            results['b'] = experiment_b_ln_compression(
                model, tokenizer, device, model_info, args.n_test)
        
        if args.exp in ["c", "all"]:
            results['c'] = experiment_c_l0_control(
                model, tokenizer, device, model_info, args.n_test)
        
        if args.exp in ["a", "all"]:
            results['a'] = experiment_a_signal_propagation(
                model, tokenizer, device, model_info, min(args.n_test, 30))
    finally:
        release_model(model)
    
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s")
    
    print("\n" + "="*70)
    print("Phase 67 综合判定")
    print("="*70)
    print("""
SCT核心预测:
1. LN对信号的压缩比<0.5
2. 信号在中层衰减最快
3. L0(β=8+)可以可靠控制输出
""")


if __name__ == "__main__":
    main()
