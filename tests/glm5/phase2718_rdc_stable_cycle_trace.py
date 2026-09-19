"""Observed complete-state trajectories versus the analytically located surrogate attractor."""
from rdc_relation_common import *


def main():
    out=BASE/'surrogate_stability';sr=read(out/'result.json');records=[]
    for cycle in sr['cycles']:
        if cycle['period']!=1 or not cycle['greedy_self_consistent'] or not cycle['locally_attracting_affine_cycle']:continue
        tid=cycle['incoming_token_cycle'][0]
        with np.load(out/f'cycles/{cycle["cycle_id"]:03d}.npz') as z:fixed=z['fixed_cycle_states'][0]
        for sid in cycle['observed_suffix_sources']:
            r=read(BASE/f'generation/commits/{sid}.json');tokens=r['self_tokens'];n=0
            for token in reversed(tokens):
                if token!=tid:break
                n+=1
            start=len(tokens)-n
            with np.load(BASE/f'generation/fields/{sid}.npz') as z:h=z['self_predicted_h36'].astype(float);native=unbits(z['native_h36_on_self_prefix']).astype(float)
            distance=np.linalg.norm(h-fixed,axis=1);norm=np.linalg.norm(h,axis=1)
            records.append({'sample_id':sid,'cycle_id':cycle['cycle_id'],'token_id':tid,'repeated_suffix_length':n,'suffix_start_step':start,'full_coordinate_distance_to_fixed_state':distance.tolist(),
              'full_coordinate_state_norm':norm.tolist(),'native_same_prefix_state_norm':np.linalg.norm(native,axis=1).tolist(),'distance_last_over_suffix_start':float(distance[-1]/max(distance[start],1e-30)),
              'suffix_distance_monotone_nonincreasing':bool(np.all(np.diff(distance[start:])<=1e-5))})
    save(out/'trajectory_result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'sources':len(records),'records':records,
      'median_last_to_suffix_start_distance_ratio':float(np.median([r['distance_last_over_suffix_start'] for r in records])) if records else None,
      'sources_with_decreasing_net_distance':sum(r['distance_last_over_suffix_start']<1 for r in records),
      'sources_with_monotone_suffix_distance':sum(r['suffix_distance_monotone_nonincreasing'] for r in records),
      'limits':'Full vectors retained in generation and cycles artifacts; these norms are complete-coordinate diagnostics, not low-rank states. A16-step approach is not proof that every generated prefix is in an invariant global basin; this is the learned surrogate, not the original LLM.'});print('STABLE_CYCLE_OBSERVED_TRAJECTORIES',len(records),flush=True)


if __name__=='__main__':main()
