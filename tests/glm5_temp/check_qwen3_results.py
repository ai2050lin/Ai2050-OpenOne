"""检查Qwen3 Phase 145结果"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np

data = json.load(open('tests/glm5_temp/phase145_qwen3_attractor_20260512_2119.json', 'r', encoding='utf-8'))
print('Keys:', list(data.keys()))
print('Model info:', data.get('model_info'))

exp_a = data.get('exp_a', {})
print(f'\nExp A entries: {len(exp_a)}')

# Summarize key metrics
for inject_l in [0, 9, 18, 27]:
    for eps in [0.5, 1.0, 2.0, 5.0]:
        random_final = []
        constraint_final = []
        for key, val in exp_a.items():
            if val["inject_layer"] == inject_l and abs(val["eps"] - eps) < 0.01:
                init_r = val["initial_dist_random"]
                init_c = val["initial_dist_constraint"]
                if init_r > 1e-10:
                    random_final.append(val["recovery_random"][-1] / init_r)
                if init_c > 1e-10:
                    constraint_final.append(val["recovery_constraint"][-1] / init_c)
        if random_final:
            print(f'  L{inject_l}, eps={eps}: random_final/init={np.mean(random_final):.3f}, constraint={np.mean(constraint_final):.3f}')

exp_c = data.get('exp_c', {})
print(f'\nExp C constraint types: {list(exp_c.keys())}')
for c_type, c_data in exp_c.items():
    traj = c_data.get('mean_delta_trajectory', [])
    if len(traj) > 1:
        peak_idx = int(np.argmax(traj))
        peak_val = traj[peak_idx]
        final_val = traj[-1]
        ratio = final_val / peak_val if peak_val > 0 else 0
        print(f'  {c_type}: peak@L{peak_idx}={peak_val:.1f}, final={final_val:.1f}, ratio={ratio:.3f}')
