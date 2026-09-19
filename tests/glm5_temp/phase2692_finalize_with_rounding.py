"""Run unchanged frozen2692 finalizer, then append the actual numerical audit.

The scalar finite-change results are NOT reclassified by baseline agreement.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'glm5'))
from phase2620_native_coordinate_contract import *

OUT = RESULT / 'phase2692_linked_native_ledger'
MARKER = '**Phase 2692 原生舍入补充结果（C007—C012，实际完成记录）**'


def main():
    phases = [int(v) for v in re.findall(r'^## Phase (\d+):', MEMO.read_text(encoding='utf-8-sig'), re.M)]
    assert phases[-1] in (2691, 2692) and phases.count(2691) == 1
    previous = read(RESULT / 'phase2691_crossmodel_role_confirmation/analysis/final.json')
    assert previous['all_checks_passed']
    from phase2693_campaign_terminal import numeric_audit
    numeric = numeric_audit()
    if phases[-1] == 2691:
        assert not (OUT / 'analysis/final.json').exists()
        import phase2692_linked_native_ledger as frozen
        rows, contract = frozen.prepare()
        assert sha(Path(frozen.__file__)) == contract['code_sha256']
        frozen.finalize(contract)
    final = read(OUT / 'analysis/final.json')
    assert final['phase'] == 2692 and final['all_checks_passed']
    if MARKER not in MEMO.read_text(encoding='utf-8-sig'):
        stamp = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
        metrics = numeric['all_stage_coordinate_metrics']
        table = ['| 数据/阶段 | 全坐标数 | 理想FP64误差L1 | 显式舍入误差L1 | 残余不一致坐标 |',
                 '|---|---:|---:|---:|---:|']
        for split in ('initial', 'confirmation'):
            for stage in ('qnorm', 'knorm', 'qrope_anchored', 'krope_anchored', 'P_anchored', 'AV_anchored', 'P_composed', 'AV_composed'):
                a = metrics[f'{split}/ideal64/{stage}']; b = metrics[f'{split}/explicit_round32/{stage}']
                table.append(f'| {split}/{stage} | {b["coordinates"]} | {a["error_L1"]:.12g} | {b["error_L1"]:.12g} | {b["mismatch_coordinates"]} |')
        text = f'\n\n{MARKER} [{stamp}]\n\n'
        text += '在原始16例与独立新填充16例的八个预定层，实际完成32例×8层的CPU只读基线核查；全为预定truth/v0，不能推广全部原始/确认条件。C007全部有限BF16位型与中点ties-to-even；C008完整Q/K归一化；C009原生归一化值输入的完整RoPE；C010原生RoPE输入的完整来源P；C011原生P/V输入的AV；C012原生线性投影起点的组合norm→RoPE→P→AV。新模型前向和参数干预均为零。\n\n'
        text += r'$$' + '\n' + r'N_B(z)=R_B\{\gamma R_B[z\,\operatorname{rsqrt}(\operatorname{mean}_{32}(z^2)+\epsilon)]\},\quad R_B=\operatorname{BF16}_{ties\text{-}to\text{-}even};\quad E_1=\sum_i|\widehat x_i-x_i^{native}|.' + '\n$$\n\n'
        text += '\n'.join(table)
        text += '\n\n全部65,280有限位型固定点及65,278中点检查通过。以原生归一化值输入的Q/K RoPE共92,569,600坐标完全一致，但归一化、P/AV及组合路径仍有残余误差。完整逐层/例数据、误差图、方法与源文件SHA在`numerical_baseline_audit/result.json`、`maps/`、`protocol.json`；正式工具`tests/glm5/phase2692_native_rounding_math.py`，实际核查`tests/glm5_temp/phase2692_rounding_baseline_audit.py`。\n\n'
        text += '理论进展是测量校准：遗漏的逐步舍入能解释所测基线的相当部分偏差，不是语言编码规律。NumPy FP32参考非CUDA内核仿真；两类归约、rsqrt、乘加及padding仍可能不同。基线误差小不能推出2689有限参数变化预测通过，尤其不能把已舍入投影加理想差项当作真实改权重后的GEMM。下一同目标任务须从完整真实W行、实际有效改变量及同协议自然生成测量，区分原生舍入与真正条件耦合，并匹配绝对/相对剂量。所有低值、零、相消及失败字段保留；全局同号/必要性不是路线唯一门。\n'
        with MEMO.open('a', encoding='utf-8') as stream:
            stream.write(text)
    save(OUT / 'analysis/numerical_review_append.json', {
        'actual_phase2692_complete': True, 'numerical_result_sha256': sha(OUT / 'numerical_baseline_audit/result.json'),
        'baseline_only_not_finite_intervention_prediction': True, 'memo_marker': MARKER,
        'wrapper_sha256': sha(Path(__file__)), 'timestamp': datetime.now().astimezone().isoformat()})
    print('2692 unchanged frozen finalizer and actual baseline-numerical supplement complete', flush=True)


if __name__ == '__main__': main()
