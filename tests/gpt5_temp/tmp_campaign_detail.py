import os, collections
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_campaign_detail.txt"
res=[]
GLM = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"

# For each campaign dir, break down by extension and by subdir depth-3
CAMPAIGNS = ["rdc_query_construction_20260913","rdc_continuity_campaign_20260909",
 "rdc_conditional_campaign_20260910","client_visualization_assets","rdc_feature_campaign_20260909",
 "rdc_law_campaign_20260911","rdc_binding_campaign_20260912","rdc_query_campaign_20260913",
 "rdc_update_campaign_20260913","rdc_mechanism_campaign_20260909","rdc_operator_atlas_20260911",
 "rdc_joint_atlas_20260911","rdc_prefix_atlas_20260910","rdc_relation_stability_20260923",
 "rdc_context_interaction_20260923","rdc_trusted_rebuild_20260923","rdc_early_interaction_20260923"]

for c in CAMPAIGNS:
    base=os.path.join(GLM,c)
    if not os.path.isdir(base): continue
    ext=collections.defaultdict(lambda:[0,0])
    sub=collections.defaultdict(lambda:[0,0])
    for dp,dn,fns in os.walk(base):
        rel=os.path.relpath(dp,base).split(os.sep)
        key = rel[0] if rel[0]!="." else "(root)"
        for fn in fns:
            fp=os.path.join(dp,fn)
            try: sz=os.path.getsize(fp)
            except OSError: continue
            e=os.path.splitext(fn)[1].lower() or "(noext)"
            ext[e][0]+=sz; ext[e][1]+=1
            sub[key][0]+=sz; sub[key][1]+=1
    tot=sum(v[0] for v in ext.values())
    res.append("\n===== %s : %.2f GB =====" % (c, tot/1024**3))
    res.append("  by ext: " + "  ".join("%s=%.1fMB(%d)"%(k,v[0]/1024**2,v[1]) for k,v in sorted(ext.items(),key=lambda x:-x[1][0])[:8]))
    res.append("  by subdir (top 12):")
    for k,(sz,cc) in sorted(sub.items(), key=lambda x:-x[1][0])[:12]:
        res.append("     %9.1f MB %7d  %s"%(sz/1024**2,cc,k))

# rdc_query_construction: top-level non-phase dirs
res.append("\n\n===== rdc_query_construction: capture breakdown =====")
cap=os.path.join(GLM,"rdc_query_construction_20260913","capture")
if os.path.isdir(cap):
    for dp,dn,fns in os.walk(cap):
        sz=sum(os.path.getsize(os.path.join(dp,f)) for f in fns if os.path.exists(os.path.join(dp,f)))
        if sz>50*1024**2:
            res.append("  %9.1f MB %6d  %s"%(sz/1024**2,len(fns),os.path.relpath(dp,cap)))

# tests/result (gpt5 line) breakdown
res.append("\n\n===== tests/result (gpt5 line) level-1 =====")
R2=r"D:\AI2050\Ai2050-OpenOne\tests\result"
if os.path.isdir(R2):
    sub=collections.defaultdict(lambda:[0,0])
    for dp,dn,fns in os.walk(R2):
        rel=os.path.relpath(dp,R2).split(os.sep)
        for fn in fns:
            try: sz=os.path.getsize(os.path.join(dp,fn))
            except OSError: continue
            d=sub[rel[0] if rel[0]!="." else "(root)"]; d[0]+=sz; d[1]+=1
    for k,(sz,cc) in sorted(sub.items(), key=lambda x:-x[1][0])[:30]:
        res.append("  %9.1f MB %7d  %s"%(sz/1024**2,cc,k))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok")
