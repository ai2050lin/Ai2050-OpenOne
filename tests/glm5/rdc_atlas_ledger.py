# -*- coding: utf-8 -*-
"""rdc_atlas_ledger.py - loader/validator for the TMA Atlas Ledger (v1/v2).

v2 additions (phase 2882, backward compatible with v1 files):
    quadrants          - channel-dissociation quadrant table (per family)
    transfer           - cross-family transfer matrix (from joint spectrum)
    negatives          - real-negative / quarantine registry
    model_namespace    - primary model + pending replication models
    migration_history  - format-version migration log

Usage:
    from rdc_atlas_ledger import AtlasLedger
    led = AtlasLedger.load()                 # validates schema + SHAs
    led.growth_table()                       # growth curve as sorted rows
    led.blocks_of('class')                   # coordinate blocks for an axis
    led.headset('class43')                   # head set with definition
    led.quadrant_rows()                      # v2 quadrant table
    led.add_measurement({...})               # append-only write + backup
"""
import hashlib
import io
import json
import os
import shutil

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
RESULT_BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
               r'\rdc_query_construction_20260913')

REQUIRED = ['format', 'version', 'model', 'axes', 'blocks', 'headsets',
            'measurements', 'growth_curve', 'linkage']
V2_OPTIONAL = ['quadrants', 'transfer', 'negatives', 'model_namespace',
               'migration_history']
KINDS = {'unembed_proj', 'causal_spectrum', 'causal_spectrum_sparse',
         'mlp_response', 'ov_static', 'eta2', 'share'}


def _sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


class AtlasLedger(object):

    def __init__(self, doc):
        self.doc = doc

    # ---------- IO ----------
    @classmethod
    def load(cls, path=LEDGER, verify_sha=True):
        with io.open(path, encoding='utf-8') as f:
            doc = json.load(f)
        missing = [k for k in REQUIRED if k not in doc]
        if missing:
            raise ValueError('ledger missing keys: %s' % missing)
        # v2 optional sections: default empty so v1 files load unchanged
        for k in V2_OPTIONAL:
            doc.setdefault(k, {} if k in ('transfer', 'model_namespace')
                           else [])
        led = cls(doc)
        if verify_sha:
            led.verify()
        return led

    def verify(self, report=None):
        """Recompute sha256_8 for every file-level src; mark stale."""
        stale = []
        for sect, items in (('blocks', self.doc['blocks']),
                            ('headsets', self.doc['headsets']),
                            ('measurements', self.doc['measurements'])):
            for it in items:
                src = it.get('src') or it.get('source') or {}
                p, want = src.get('path'), src.get('sha256_8')
                if not p or not want:
                    continue
                full = os.path.join(RESULT_BASE, p.replace('/', os.sep))
                if not os.path.exists(full):
                    it['stale'] = 'file_missing'
                    stale.append((sect, it.get('block_id')
                                  or it.get('meas_id')
                                  or it.get('set_id'), 'file_missing'))
                elif _sha8(full) != want:
                    it['stale'] = 'sha_mismatch'
                    stale.append((sect, it.get('block_id')
                                  or it.get('meas_id')
                                  or it.get('set_id'), 'sha_mismatch'))
        if report is not None:
            report.extend(stale)
        return stale

    def save(self, path=LEDGER, backup=True):
        if backup and os.path.exists(path):
            shutil.copy2(path, path + '.bak')
        with io.open(path, 'w', encoding='utf-8') as f:
            json.dump(self.doc, f, indent=2, ensure_ascii=False)

    # ---------- accessors ----------
    def blocks_of(self, axis_id):
        return [b for b in self.doc['blocks'] if b['axis_id'] == axis_id]

    def headset(self, set_id):
        for h in self.doc['headsets']:
            if h['set_id'] == set_id:
                return h
        raise KeyError(set_id)

    def quadrant_rows(self):
        """v2: channel-dissociation quadrant rows (family-ordered)."""
        return list(self.doc.get('quadrants', []))

    def negatives(self):
        """v2: real-negative / quarantine registry."""
        return list(self.doc.get('negatives', []))

    def transfer_matrix(self):
        """v2: cross-family transfer block (dict, may be empty in v1)."""
        return dict(self.doc.get('transfer', {}))

    def block_npz(self, block_id):
        for b in self.doc['blocks']:
            if b['block_id'] == block_id:
                full = os.path.join(RESULT_BASE,
                                    b['src']['path'].replace('/', os.sep))
                import numpy as np
                return np.load(full, allow_pickle=True)[b['src']['key']]
        raise KeyError(block_id)

    def growth_table(self):
        rows = sorted(self.doc['growth_curve'],
                      key=lambda r: (r.get('axis_id', ''), r['phase']))
        return ['%-24s %-16s comp=%-5d acc=%-7s new=%s'
                % (r['point_id'], r.get('axis_id', '-'),
                   r.get('components', -1), r.get('acc', '-'),
                   r.get('new', '-')) for r in rows]

    # ---------- append-only writers ----------
    def add_measurement(self, meas):
        need = {'meas_id', 'type', 'verdict', 'source'}
        if not need <= set(meas):
            raise ValueError('measurement needs %s' % sorted(need))
        if any(m['meas_id'] == meas['meas_id']
               for m in self.doc['measurements']):
            raise ValueError('meas_id exists: %s' % meas['meas_id'])
        self.doc['measurements'].append(meas)

    def add_block(self, blk):
        need = {'block_id', 'axis_id', 'kind', 'shape', 'src'}
        if not need <= set(blk):
            raise ValueError('block needs %s' % sorted(need))
        if blk['kind'] not in KINDS:
            raise ValueError('unknown kind: %s' % blk['kind'])
        if any(b['block_id'] == blk['block_id']
               for b in self.doc['blocks']):
            raise ValueError('block_id exists: %s' % blk['block_id'])
        self.doc['blocks'].append(blk)


if __name__ == '__main__':
    rep = []
    led = AtlasLedger.load(verify_sha=True)
    print('stale refs:', rep if led.verify(rep) else 'none')
    print('\n'.join(led.growth_table()))
