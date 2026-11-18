"""Immutable B18-B28 native atlas transport, using the audited streaming installer."""
import argparse
from pathlib import Path
if __package__:
    from . import qwen14_mlp_pack as transport
else:
    import qwen14_mlp_pack as transport

SPECS={
 'QWEN4-GOODFIRE-B18-B28-20261113-v2':(9728,2560,'6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad'),
 'QWEN14-GOODFIRE-B18-B28-20261113-v2':(17408,5120,'1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce')}


def validate(d):
    identity=SPECS.get(d.get('run_id'))
    if identity is None or (d.get('width'),d.get('hidden_size'),d.get('model_snapshot'))!=identity:
        raise ValueError('Wrong native model, dimensions or run')
    if (d.get('schema'),d.get('state'),d.get('kind'),d.get('blocks'))!=('native_mlp_manifest.v1','complete','real',list(range(18,29))):
        raise ValueError('Incomplete or wrong native atlas')
    files={}
    for r in d['files']:
        n=r['path']
        if not isinstance(n,str) or Path(n).name!=n or n in ('.','..','manifest.json') or ':' in n or '\\' in n or n in files:
            raise ValueError('Unsafe or repeated filename')
        if type(r['bytes']) is not int or not 0<r['bytes']<=4*2**30:raise ValueError('Invalid file size')
        if len(r['sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['sha256']):raise ValueError('Invalid checksum')
        if 'source_run' in r:raise ValueError('Middle atlas must be self-contained')
        files[n]=r
    required={'report.json','audit.json','documents.json','tokenizer.json','prefix-extrema.npz','QWEN-LICENSE.txt','WEIGHT-NOTICE.txt',
              'natural_exact-effects.json','capability_exact-effects.json','capability-inputs.json'}
    required.update(f'B{b}-{s}.npy' for b in d['blocks'] for s in ['G','V','A','input','W-gate','W-up','W-down'])
    required.update(f'B{b}-{s}.json' for b in d['blocks'] for s in ['profile','cues','readout'])
    required.update(f'B{b}-{s}.npz' for b in d['blocks'] for s in ['regimes','prediction'])
    required.update(f'B{b}-B{b+1}-prediction.json' for b in d['blocks'][:-1])
    if not required<=files.keys():raise ValueError('Missing required native files')
    return files


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--out',required=True)
    v=sub.add_parser('verify');v.add_argument('--archive',required=True)
    i=sub.add_parser('import');i.add_argument('--archive',required=True);i.add_argument('--data-dir',required=True)
    a=p.parse_args();transport.validate=validate
    try:
        if a.action=='build':print(transport.build(a.source,a.out))
        elif a.action=='verify':print({'verified_manifest_sha256':transport.stream_check(a.archive)[1]})
        else:print(transport.install(a.archive,a.data_dir))
    except transport.StorageBlocked as error:transport.stop_with_notice(error)


if __name__=='__main__':main()
