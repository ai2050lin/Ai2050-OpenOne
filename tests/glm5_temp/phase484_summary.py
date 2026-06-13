"""Phase 484 Results Summary"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

SEP = "=" * 70

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase484_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n{SEP}')
    print(f'  {model.upper()} - Phase 484 Results')
    print(f'{SEP}')
    
    # Exp1
    print(f'\n--- Exp1: Writer Reconstruction ---')
    for cat, d in data.get('exp1_writer_reconstruction', {}).items():
        cos_k = d.get('cos_at_k', {})
        energy_k = d.get('energy_at_k', {})
        n_sig = d.get('neuron_contribution_stats', {}).get('n_significant', 0)
        total = d.get('neuron_contribution_stats', {}).get('total_neurons', 0)
        print(f'  {cat}: cos@10={cos_k.get("10",0):.3f}, cos@50={cos_k.get("50",0):.3f}, '
              f'cos@200={cos_k.get("200",0):.3f}, energy@50={energy_k.get("50",0):.3f}, '
              f'significant={n_sig}/{total}, cos_diff_y={d.get("cos_diff_y",0):.3f}')
    
    # Exp2
    print(f'\n--- Exp2: Causal Test ---')
    for cat, d in data.get('exp2_writer_causal_test', {}).items():
        dir_target = d.get('direction_remove_target', 0)
        ablate = d.get('ablation_results', {})
        for k_name in ['k=5', 'k=10', 'k=20', 'k=50']:
            if k_name in ablate:
                kd = ablate[k_name]
                print(f'  {cat}/{k_name}: target_D={kd["target_delta"]:.2f}, cos_remove={kd["cos_with_direction_remove"]:.3f}')
    
    # Exp3
    print(f'\n--- Exp3: Relation Slot Readout ---')
    for cat, d in data.get('exp3_relation_slot_readout', {}).items():
        rels = d.get('relations', {})
        print(f'  {cat}:')
        for rel, rd in rels.items():
            s1 = rd.get('scale_1.0', {})
            bc_sel = s1.get('Bc_inject_selectivity', 0)
            mc_sel = s1.get('Mc_inject_selectivity', 0)
            bc_target = s1.get('Bc_inject_target_delta', 0)
            mc_target = s1.get('Mc_inject_target_delta', 0)
            print(f'    {rel}: Bc_sel={bc_sel:.2f}(D={bc_target:.2f}), Mc_sel={mc_sel:.2f}(D={mc_target:.2f})')
    
    # Exp4
    print(f'\n--- Exp4: Anomalous Competition ---')
    for pair, d in data.get('exp4_anomalous_competition', {}).items():
        cos_b = d.get('cos_between_boundaries', 0)
        attrs = d.get('shared_attribute_analysis', {})
        print(f'  {pair}: cos(boundaries)={cos_b:.3f}')
        for attr, changes in attrs.items():
            mean_change = sum(changes.values()) / len(changes) if changes else 0
            top_word = max(changes, key=changes.get) if changes else 'none'
            top_val = changes.get(top_word, 0)
            print(f'    {attr}: mean_change={mean_change:.4f}, top={top_word}({top_val:.4f})')
