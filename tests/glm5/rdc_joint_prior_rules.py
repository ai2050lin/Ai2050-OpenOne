"""Read-only exact reconstruction of Phase2718 source-RMS fits and a predeclared history query proposal."""
import hashlib
from rdc_joint_common import *
from rdc_relation_common import rows as old_material, load_field as old_field
from rdc_relation_estimators import Bank, select, predict
from rdc_relation_inference import CurrentRule
from rdc_relation_native_parameters import parameter, decode


def rms_sources(h):
    h = np.asarray(h, np.float32)
    scale = np.sqrt(np.mean(h.astype(float)**2, axis=1))
    return (h / np.maximum(scale[:, None], 1e-8)).astype(np.float32)


def array_hashes(model):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in model.items()}


class PriorRMS:
    def __init__(self):
        data = {'current': [], 'history_mean': []}
        targets, meta = [], []
        for fresh in (False, True):
            for row in old_material(fresh):
                z = old_field(row, fresh)
                h12, h23, h36 = (unbits(z[k]) for k in ('h12', 'h23', 'h36'))
                for j, p in enumerate(row['anchors']):
                    data['current'].append(h12[p])
                    data['history_mean'].append(rms_sources(h12[:p]).mean(0))
                    targets.append(np.concatenate([h23[p], h36[3*j]]))
                    meta.append('fresh' if fresh else row['split'])
        data = {k: np.stack(v).astype(np.float32) for k, v in data.items()}
        targets = np.stack(targets).astype(np.float32)
        tr = np.array([i for i, s in enumerate(meta) if s == 'train'])
        va = np.array([i for i, s in enumerate(meta) if s == 'validation'])
        bank = Bank(data, tr)
        dots = bank.dots()
        self.train = {k: v[tr] for k, v in data.items()}
        self.scales = bank.scales
        self.fits = {}
        receipts = {}
        expected = read(PREVIOUS / 'source_normalization/reconstructible_fits.json')
        df = read(PREVIOUS / 'rules/current/result.json')['effective_df']
        for name, fixed_df in [('source_RMS_mean', None), ('source_RMS_mean_matched_df', df)]:
            model, info, _ = select(dots, 'history_mean', targets, tr, va, [(0, 2560), (2560, 5120)], fixed_df=fixed_df)
            actual = array_hashes(model)
            assert actual == expected[name]['array_sha'], ('Old RMS exact reconstruction failed', name, actual, expected[name]['array_sha'])
            self.fits[name] = model, info
            receipts[name] = {'selection': info, 'coefficient_sha': actual, 'all_old_coefficient_hashes_equal': True}
        self.linear = CurrentRule('current')
        self.quad = CurrentRule('full_quadratic')
        report = {'timestamp': stamp(), 'source': snapshot(Path(__file__)), 'old_fits': receipts,
                  'old_fit_manifest_sha': sha(PREVIOUS / 'source_normalization/reconstructible_fits.json'),
                  'new_data_used_for_fitting_or_selection': False}
        save(BASE / 'prior_confirmation/reconstruction.json', report)

    def __call__(self, current, mean):
        query = Bank({'current': np.asarray(current, np.float32), 'history_mean': np.asarray(mean, np.float32)}, scales=self.scales)
        train = Bank(self.train, scales=self.scales)
        dots = query.dots(train)
        output = {'current': self.linear(current), 'full_quadratic': self.quad(current)}
        for name, (model, info) in self.fits.items():
            output[name] = predict(model, Bank.gram(dots, 'history_mean', info['mix']))
        return output


class QueryProposal:
    """Old-training-only H36(previous), known E(next) -> H12(next); never a target-state feature."""
    def __init__(self):
        table = parameter(ROOT, 'model.embed_tokens.weight')
        h, e, target, meta = [], [], [], []
        for row in old_material():
            if row['split'] not in ('train', 'validation'):
                continue
            z = old_field(row)
            for j, p in enumerate(row['anchors']):
                h.append(unbits(z['h36'][3*j]))
                e.append(decode(table[row['prompt_ids'][p+1]]))
                target.append(unbits(z['h12'][p+1]))
                meta.append(row['split'])
        h, e, y = np.stack(h).astype(np.float32), np.stack(e).astype(np.float32), np.stack(target).astype(np.float32)
        tr = np.array([i for i, s in enumerate(meta) if s == 'train'])
        va = np.array([i for i, s in enumerate(meta) if s == 'validation'])
        self.hscale = float(np.mean(np.sum(h[tr].astype(float)**2, 1)))
        self.escale = float(np.mean(np.sum(e[tr].astype(float)**2, 1)))
        hd = h.astype(float) @ h.astype(float).T / self.hscale
        ed = e.astype(float) @ e.astype(float).T / self.escale
        dots = {'current': hd, 'history_mean': ed + hd*ed}
        self.model, self.info, grid = select(dots, 'history_mean', y, tr, va, [(0, 2560)])
        self.htrain, self.etrain = h[tr].astype(float), e[tr].astype(float)
        prediction = predict(self.model, Bank.gram(dots, 'history_mean', self.info['mix'])[np.ix_(va, tr)])
        receipt = {'source': snapshot(Path(__file__)), 'selection': self.info, 'grid': grid,
                   'train_anchors': len(tr), 'validation_anchors': len(va), 'training_scope': 'Only320old training sources;96old validation selects. No new main/fresh targets.',
                   'target': 'Next H12, not next output token. New token embedding is already known.',
                   'validation_MSE': float(np.mean((prediction-y[va])**2)),
                   'coefficient_sha': array_hashes(self.model), 'hscale': self.hscale, 'escale': self.escale,
                   'retention': 'Exact fitting recipe plus old immutable fields; no coordinate truncation or duplicated large coefficient file.'}
        path = BASE / 'query_proposal/frozen.json'
        if path.exists():
            prior = read(path)
            assert prior['coefficient_sha'] == receipt['coefficient_sha'] and prior['selection'] == receipt['selection']
        else:
            save(path, {'timestamp': stamp(), **receipt})
        print('JOINT_QUERY_PROPOSAL', self.info, 'validation MSE', receipt['validation_MSE'], flush=True)

    def __call__(self, previous, embedding):
        h = np.atleast_2d(previous).astype(float) @ self.htrain.T / self.hscale
        e = np.atleast_2d(embedding).astype(float) @ self.etrain.T / self.escale
        return predict(self.model, Bank.gram({'current': h, 'history_mean': e+h*e}, 'history_mean', self.info['mix']))
