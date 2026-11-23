"""Audit display taxonomy without changing frozen grammar; retain prior client assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import checked_dir,check_write
from shared.dist_protocol import pieces,lang_map


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path)
    args=parser.parse_args()
    out=checked_dir(ROOT/'tests/glm5/result/source_binding_mapping_20261119',10*2**20)
    temp=checked_dir(ROOT/'tests/glm5_temp/source_binding_mapping_20261119',200*2**20)
    previous=ROOT/'release/ai2050-center-1.0.41.tar.gz'
    assert hashlib.sha256(previous.read_bytes()).hexdigest()==previous.with_name(previous.name+'.sha256').read_text().split()[0]
    prefix='ai2050-center-1.0.41/'
    grammar='shared/dist_protocol/grammars/SOURCE-BINDING-20261110.json'
    tree_path='shared/dist_protocol/lang_tree.json'
    tree=pieces.load_tree()
    assert not pieces.validate_tree(tree)
    modern=lang_map.load()
    catalog=json.loads((ROOT/'shared/dist_protocol/lang_catalog.json').read_text(encoding='utf-8'))
    assert not lang_map.validate(modern['map'],modern['words'],catalog)
    with tarfile.open(previous) as archive:
        old_tree=json.load(archive.extractfile(prefix+tree_path))
        old_grammar=archive.extractfile(prefix+grammar).read()
        old_manifest=json.load(archive.extractfile(prefix+'RELEASE_MANIFEST.json'))
        assert old_grammar==(ROOT/grammar).read_bytes(),'frozen grammar changed'
        assert tree['version']>old_tree['version']
        # Other families, labels, and taxonomy objects retain their meaning and stable IDs.
        for key in ('family_map','family_languages'):
            for identity,value in old_tree[key].items():assert tree[key][identity]==value
        for key in ('types','categories','items'):
            current={v['id']:v for v in tree[key]}
            for value in old_tree[key]:assert current[value['id']]==value
        g=json.loads(old_grammar)
        resolved={f:pieces.factor_item(tree,g['family_id'],f,g.get('lang_items')) for f in g['factors']}
        assert set(resolved.values())=={'KN.SOURCE','KN.ENTITY_REF','EXT.SOURCE_ORDER','EXT.SOURCE_WORDING'}
        compatible={f:modern['map']['tree_compat'][old] for f,old in resolved.items()}
        modern_nodes={item['id']:item for d in modern['map']['domains'] for sub in d['subs'] for item in sub['items']}
        modern_nodes.update({item['id']:item for item in modern['map']['dims']})
        for node_id in compatible.values():assert g['family_id'] in modern_nodes[node_id]['templates']
        assets={k:v for k,v in old_manifest['files'].items() if k.startswith('frontend/dist/')}
        if not args.package:
            for relative,value in assets.items():
                name=PurePosixPath(relative)
                assert not name.is_absolute() and '..' not in name.parts
                raw=archive.extractfile(prefix+relative).read()
                assert hashlib.sha256(raw).hexdigest()==value
                dest=check_write(temp/'prior-client'/Path(*name.parts[2:]),len(raw),operation='复用已验收客户端')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    result={'grammar_sha256':hashlib.sha256(old_grammar).hexdigest(),'frozen_grammar_unchanged':True,
            'tree_version_before':old_tree['version'],'tree_version_after':tree['version'],
            'factor_mapping':resolved,'existing_taxonomy_unchanged':True,'family_language':tree['family_languages'][g['family_id']],
            'client_compat_mapping':compatible,'client_map_manifest':modern['manifest']['files'],
            'client_source_version':'1.0.41','client_assets':len(assets),'new_package_verified':False}
    if args.package:
        version=(ROOT/'VERSION').read_text().strip();new_prefix='ai2050-center-'+version+'/'
        with tarfile.open(args.package) as archive:
            manifest=json.load(archive.extractfile(new_prefix+'RELEASE_MANIFEST.json'))
            packaged_assets={k:v for k,v in manifest['files'].items() if k.startswith('frontend/dist/')}
            assert packaged_assets==assets,'client assets differ from previously verified release'
            assert archive.extractfile(new_prefix+grammar).read()==old_grammar
            assert archive.extractfile(new_prefix+tree_path).read()==(ROOT/tree_path).read_bytes()
            for path in ['shared/dist_protocol/lang_catalog.json',*[f'shared/dist_protocol/lang_map/{name}' for name in ['structure.json','lang_map.json','words.json','manifest.json']]]:
                assert archive.extractfile(new_prefix+path).read()==(ROOT/path).read_bytes()
            assert archive.extractfile(new_prefix+'VERSION').read().decode().strip()==version
        result.update(new_package_verified=True,version=version,package=str(args.package))
    check_write(out/'mapping-audit.json',10000)
    (out/'mapping-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
