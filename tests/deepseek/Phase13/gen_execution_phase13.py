# -*- coding: utf-8 -*-
"""Phase 13 execution 冻结件生成器（N2h1-alpha-6）。

exec 记录：臂定义 / 坐标 / bootstrap / G 族 / floors / 继承锚（含 seal sha）。
本 exec 在**正式运行前**生成并冻结；主脚本会断言 REQUIRED 字段齐全。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T12 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


R12 = os.path.join(T12, 'result_phase12.json')
E12 = os.path.join(T12, 'execution_phase12.json')
SEAL = os.path.join(T13, 'N2h1a6_design_seal.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')

R = json.load(io.open(R12, encoding='utf-8'))
E5ROWS = R['E5']['7']['rows']
CONF_ALPHAS = [r['alpha'] for r in E5ROWS]

EXEC = {
    "phase": 13,
    "name": "N2h1-alpha-6 / pairwise-site paired bootstrap",

    "zero_extra_forward": True,
    "cpu_only": True,
    "no_torch": True,

    "seed": 20261001,

    "source": {
        "phase12_result_path": "tests/deepseek_temp/Phase12/result_phase12.json",
        "phase12_result_sha256": sha(R12),
        "phase12_exec_path": "tests/deepseek_temp/Phase12/execution_phase12.json",
        "phase12_exec_sha256": sha(E12),
        "memo_sha256_pre_freeze": sha(MEMO),
    },

    "seal_path": "tests/deepseek_temp/Phase13/N2h1a6_design_seal.json",
    "seal_sha256": sha(SEAL),
    "seal_sha8": sha(SEAL)[:8],

    "sites": {
        "profile": [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
        "confirmation": [7, 11, 20, 34],
        "n_adjacent_pairs": 17,
        "n_conf_pairs": 3,
    },

    "alpha_grid": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    "conf_alpha_grid": CONF_ALPHAS,
    "window_W": 3,
    "jdose_floor": 0.01,
    "xh_frac": 0.5,

    "bootstrap": {
        "B": 2000,
        "B_perm": 2000,
        "seed": 20261001,
        "ci": [2.5, 97.5],
        "replay_key": "BRNG 的首个消费点即主循环 integers(0,nP,nP)，故 A0/A0b 可逐位复现",
        "conf_denominator": "FULL_SWAP (constant; 逐字沿用 Phase 12 确认集口径)",
        "disc_denominator": "FS_VEC[idx].mean() (per-pair, 逐字沿用 Phase 12)",
    },

    "arms": {
        "A0_replicate_phase12": {"smoke_bs": 200, "assert_max_abs_dev": 0.0},
        "A0b_replicate_confirmation_band": {"assert_max_abs_dev": 1e-12},
        "A1_delta_J": {"F": "J_only"},
        "A2_delta_xhalf": {"F": "cross_alpha"},
        "A3_tightening_attribution": {"ddof": 0},
        "A4_concentration_tail_probabilities": {"thresholds": [0.60, 0.40]},
        "A5_independent_vs_paired_count": {"indep_rule": "J_ci band disjoint"},
        "A6_deep_tail_reversal": {"pairs": [[28, 30], [30, 32], [32, 34]]},
        "A7_confirmation_paired": {"n_sites": 4},
        "A8_alpha_grid_leave_one_out": {"keep": [0.0, 1.0]},
        "A9_steepness_statistic_robustness": {"alt_denominator": "IQR"},
    },

    "decision": {
        "G0p": "A0 max|d|==0 AND A0b max|d|<1e-12",
        "disc_verdict": "PAIRED_TEST_UNINFORMATIVE if N_dec_J==0 and N_dec_X==0",
        "coord_dep_rule": "abs(share_x)>=0.40 AND abs(share_j)>=0.40 AND abs(W_x-W_j)>=3",
        "P_few_threshold": 0.95,
        "P_acc_threshold": 0.95,
        "verdict_table": [
            ["G0p_fail", "DEVICE_ANCHOR_FAILED"],
            ["N_dec_J==0 and N_dec_X==0", "PAIRED_TEST_UNINFORMATIVE"],
            ["coord_dep_rule", "CONCENTRATION_COORDINATE_DEPENDENT"],
            ["P_few_x>=0.95 and P_few_j>=0.95", "CONCENTRATION_FEW_LAYER_ROBUST"],
            ["P_acc_x>=0.95 and P_acc_j>=0.95", "CONCENTRATION_ACCUMULATE_ROBUST"],
            ["else", "CONCENTRATION_UNDECIDED"],
        ],
    },

    "floors": ["F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23"],

    "result_keys": [
        "phase", "name", "smoke", "elapsed_s", "zero_extra_forward", "torch_imported",
        "panel", "anchors", "A0_replicate", "A0b_replicate_conf", "A1_delta_J", "A2_delta_xhalf",
        "A3_tightening", "A4_concentration", "A5_counts", "A6_deep_tail", "A7_confirmation",
        "A8_grid_loo", "A9_steepness_alt", "predictions_check", "floors", "P13_verdict",
        "disc_verdict", "honesty", "seal_sha8", "exec_sha8", "phase12_result_sha8",
    ],
}

OUT = os.path.join(T13, 'execution_phase13.json')
io.open(OUT, 'w', encoding='utf-8').write(json.dumps(EXEC, ensure_ascii=False, indent=1))
b = open(OUT, 'rb').read()
print('WROTE', OUT)
print('bytes', len(b), 'sha8', hashlib.sha256(b).hexdigest()[:8])
J = json.load(io.open(OUT, encoding='utf-8'))
print('readback keys', len(J.keys()), 'seal_sha8', J['seal_sha8'])
assert J['seal_sha8'] == '808c4575'
assert len(J['arms']) == 11
assert len(J['sites']['profile']) == 18
print('READBACK OK')
