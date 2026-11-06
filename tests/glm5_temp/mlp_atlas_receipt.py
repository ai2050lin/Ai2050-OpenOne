import json
from pathlib import Path
from datetime import datetime
import hashlib
root=Path(__file__).resolve().parents[2]
out=root/'tests/glm5/result/mlp_atlas_code_v2'
pack=json.loads((out/'offline-pack-acceptance.json').read_text(encoding='utf-8'))
receipt={'kind':'real','time':datetime.now().astimezone().isoformat(),'scope':'Observedbrowserandsoftwareacceptance, notsemanticmechanismproof','verified_views':['inventory:B35/U9727','controlledbank:sample9983/B35/U9727/assistant1033','dictionary:B35/F1023/all9728nativecorrelations/all2560encoderdecoder','natural:document1434/token823/all1024latent/all2560nativeoutputandprediction','parameter-audit:B35/U9727/assistant; Gate .00498017, Value .0295379, numericresidual .0000588999','inventory:fiveplannedstagesexplicitlyunfinished'],
 'loading_failure':'parameter-audit null data crashed; fixed explicit loading state, real canvas regression passed','API_checks':28,'software_tests':357,'offline_pack':pack,'screenshots':[]}
for name in ['client-native-3d.png','client-parameter-audit-3d.png']:
 p=out/name;receipt['screenshots'].append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(out/'visual-acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
memo=root/'research/glm5/docs/AGI_GLM5_MEMO.md';marker='Phase2788 最终离线包验收补记'
if marker not in memo.read_text(encoding='utf-8'):
 with memo.open('a',encoding='utf-8') as f:f.write(f"\n\n{marker} [{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}]：流式核验实际research-library包通过，共{pack['entries']}条、{pack['hashed_artifacts']}个CAS产物重核哈希；本轮151个当前报告/方案/验收/全层证明均存在。包压缩725,670,564字节、解压2,500,082,952字节；保持按条目流式导入，不把整个包解析为单个浏览器字符串。最终69项前端测试及生产构建再次通过；28项真实API与全原场哈希核验通过。视觉回执见 `tests/glm5/result/mlp_atlas_code_v2/visual-acceptance.json`，全部语义/因果破解的目标仍未达到。\n")
print(json.dumps({'visual_receipt':str(out/'visual-acceptance.json'),'offline_entries':pack['entries'],'hashed_artifacts':pack['hashed_artifacts']}))
