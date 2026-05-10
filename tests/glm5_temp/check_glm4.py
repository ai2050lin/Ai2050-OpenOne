import sys, json
sys.stdout.reconfigure(encoding='utf-8')
with open('tests/glm5_temp/phase97_exp3_glm4_primitive_decomposition.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看狗的详细轨迹
for traj in data['trajectories'][:2]:
    prompt = traj['prompt']
    en_target = traj['en_target']
    zh_source = traj['zh_source']
    print(f"=== {prompt} -> {en_target} (zh={zh_source}) ===")
    for l in sorted(traj['layers'].keys(), key=int):
        ld = traj['layers'][l]
        en_p = ld['en_target_prob']
        zh_p = ld['zh_source_prob']
        top1 = ld['top5_tokens'][0]
        top1_p = ld['top5_probs'][0]
        if int(l) >= 28 or en_p > 0.001 or zh_p > 0.01:
            print(f"  L{l}: en={en_p:.6f}, zh={zh_p:.6f}, top1={top1}({top1_p:.4f})")
    print()
