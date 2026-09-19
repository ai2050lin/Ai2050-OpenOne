import io

p = 'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase2846_fullhead_census.py'
src = open(p, encoding='utf-8').read()

old = """            # attribution (free): direct cdir write per head
            for h in range(NH):
                kv = h // group
                vin_same = sain2['same'][li][:2]
                v_same = vproj[li](torch.tensor(
                    vin_same, device='cuda',
                    dtype=next(vproj[li].parameters()).dtype)
                ).detach().float().cpu().numpy().astype(np.float64)
                Vh = v_same.reshape(2, n_kv, hd)[:, kv, :]
                Wm = Vh @ OV[(li, h)].T
                a_same = aw2['same'][li][h][1, :2]
                Cs = a_same[:, None] * Wm
                vin_f = sain2['func'][li][:2]
                v_f = vproj[li](torch.tensor(
                    vin_f, device='cuda',
                    dtype=next(vproj[li].parameters()).dtype)
                ).detach().float().cpu().numpy().astype(np.float64)
                Cf = aw2['func'][li][h][1, :2][:, None]
                    * (v_f.reshape(2, n_kv, hd)[:, kv, :]
                       @ OV[(li, h)].T)
                vin_n = sain2['null'][li][:2]
                v_n = vproj[li](torch.tensor(
                    vin_n, device='cuda',
                    dtype=next(vproj[li].parameters()).dtype)
                ).detach().float().cpu().numpy().astype(np.float64)
                Cn = aw2['null'][li][h][1, :2][:, None]
                    * (v_n.reshape(2, n_kv, hd)[:, kv, :]
                       @ OV[(li, h)].T)
                ssp = Cs - 0.5 * (Cf + Cn)
                s0L[i, h] = float(ssp[0] @ cdir)
                s1L[i, h] = float(ssp[1] @ cdir)"""

new = """            # attribution (free): direct cdir write per head
            vdt = next(vproj[li].parameters()).dtype
            Vc = {}
            for cn in c2:
                vin = sain2[cn][li][:2]
                Vc[cn] = vproj[li](torch.tensor(
                    vin, device='cuda', dtype=vdt)
                ).detach().float().cpu().numpy()
            for h in range(NH):
                kv = h // group
                s_dir = {}
                for cn in c2:
                    Vh = Vc[cn].reshape(2, n_kv, hd)[:, kv, :]
                    s_dir[cn] = aw2[cn][li][h][1, :2][:, None] \\
                        * (Vh @ OV[(li, h)].T)
                ssp = s_dir['same'] \\
                    - 0.5 * (s_dir['func'] + s_dir['null'])
                s0L[i, h] = float(ssp[0] @ cdir)
                s1L[i, h] = float(ssp[1] @ cdir)"""

# The original file has line-continuation backslashes; normalize by
# locating the block through markers instead of exact text.
start = src.find('            # attribution (free): direct cdir write per head')
end = src.find('            # causal clamp per head in this layer')
assert start != -1 and end != -1 and start < end, 'markers not found'
src = src[:start] + new + '\n\n' + src[end:]
open(p, 'w', encoding='utf-8', newline='\n').write(src)
print('patched ok')
