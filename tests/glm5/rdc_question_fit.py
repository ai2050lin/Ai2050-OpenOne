"""Exact sample-space solver for full-coordinate question/context predictors.

No native execution, target discovery, feature selection or dataset access here.
All eigenvectors are retained. Sparse lexical zeros are numerical zeros, not a
data-dependent list of selected coordinates.
"""
import numpy as np
from scipy import sparse
from rdc_question_selectivity import contrast_transform, group_center


def centered_kernel(raw, weights, column_mean=None, grand_mean=None):
    if column_mean is None:
        column_mean = weights @ raw
        grand_mean = float(column_mean @ weights)
    return raw-(raw @ weights)[:, None]-column_mean[None, :]+grand_mean, column_mean, grand_mean


class FeatureKernel:
    def __init__(self, query, context, weights, lexical=False, auxiliaries=6):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.lexical, self.auxiliaries = lexical, auxiliaries
        self.state = {}
        self.training = {}
        for role, values in [('query', query), ('context', context)]:
            if lexical:
                assert sparse.isspmatrix_csr(values)
                values = values.astype(np.float64).copy()
                width = values.shape[1]
                mean = np.asarray(self.weights @ values).ravel()
                scale = np.ones(width)
                if role == 'query':
                    aux = values[:, -auxiliaries:].toarray()
                    auxmean = self.weights @ aux
                    auxscale = np.maximum(np.sqrt(self.weights @ ((aux-auxmean)**2)), 1e-8)
                    scale[-auxiliaries:] = auxscale
                # Counts are divided by token count by the caller; never
                # standardized. Only the six known continuous fields scale.
                normalized = values.multiply(1/scale).tocsr()
                mean = mean/scale
                self.training[role] = normalized
                self.state[role+'_mean'] = mean
                self.state[role+'_scale'] = scale
            else:
                values = np.asarray(values, dtype=np.float64)
                mean = self.weights @ values
                scale = np.maximum(np.sqrt(self.weights @ ((values-mean)**2)), 1e-8)
                self.training[role] = (values-mean)/scale
                self.state[role+'_mean'] = mean
                self.state[role+'_scale'] = scale
        self.train_parts = self.parts(query, context)

    def parts(self, query, context):
        result = []
        for role, values in [('query', query), ('context', context)]:
            mean, scale = self.state[role+'_mean'], self.state[role+'_scale']
            train = self.training[role]
            if self.lexical:
                xx = values.multiply(1/scale).tocsr()
                raw = (xx @ train.T).toarray()
                raw -= np.asarray(xx @ mean)[:, None]
                raw -= np.asarray(train @ mean)[None, :]
                raw += float(mean @ mean)
            else:
                xx = (np.asarray(values, dtype=np.float64)-mean)/scale
                raw = xx @ train.T
            raw /= len(mean)
            assert np.isfinite(raw).all()
            result.append(raw)
        return result

    def kernel(self, parts, interaction):
        q, c = parts
        raw = q+c+interaction*q*c
        train_raw = self.train_parts[0]+self.train_parts[1]+interaction*self.train_parts[0]*self.train_parts[1]
        column = self.weights @ train_raw
        grand = float(column @ self.weights)
        return centered_kernel(raw, self.weights, column, grand)[0], column, grand


class Spectrum:
    def __init__(self, centered, groups, weights, strength):
        n = len(weights)
        self.b = contrast_transform(np.eye(n), groups, weights, strength)
        gram = self.b @ centered @ self.b.T
        gram = (gram+gram.T)/2
        eigenvalues, self.vectors = np.linalg.eigh(gram)
        self.negative_min = float(eigenvalues.min())
        tolerance = 1e-10*max(1., float(np.abs(eigenvalues).max()))
        assert self.negative_min >= -tolerance
        self.eigenvalues = np.maximum(eigenvalues, 0.)
        self.floor = max(1e-12, 1e-10*float(self.eigenvalues.max()))
        self.left = self.b.T @ self.vectors

    def ridge(self, target):
        ev = self.eigenvalues
        def df(value):
            return float(np.sum(ev/(ev+value)))
        attainable = df(self.floor)
        if attainable <= target:
            return self.floor, {'DF_target': int(target), 'DF_actual': attainable, 'saturated': True,
                'lambda_floor': self.floor, 'DF_at_floor': attainable}
        low, high = self.floor, max(1., float(ev.max()))
        while df(high) > target:
            high *= 2
        for _ in range(100):
            mid = (low+high)/2
            if df(mid) > target:
                low = mid
            else:
                high = mid
        value = (low+high)/2
        return value, {'DF_target': int(target), 'DF_actual': df(value), 'saturated': False,
            'lambda_floor': self.floor, 'DF_at_floor': attainable}

    def solution(self, ridge):
        return (self.left/(self.eigenvalues+ridge)[None, :]) @ self.left.T

    def predictions(self, cross, targets, weights, ridge):
        mean = weights @ targets
        transformed = self.left.T @ (targets-mean)
        predicted = ((cross @ self.left)/(self.eigenvalues+ridge)[None, :]) @ transformed+mean
        return predicted


def evaluation_arrays(predicted, actual, groups):
    n, d = actual.shape
    w = np.ones(n)/n
    pc = group_center(predicted, groups, w)
    ac = group_center(actual, groups, w)
    diff = predicted-actual
    cdiff = pc-ac
    return {'absolute_MSE_by_question': np.mean(diff**2, axis=1),
        'within_MSE_by_question': np.mean(cdiff**2, axis=1),
        'zero_change_MSE_by_question': np.mean(ac**2, axis=1),
        'predicted_change_MSE_by_question': np.mean(pc**2, axis=1),
        'response_inner_product_by_question': np.mean(pc*ac, axis=1),
        'absolute_MSE_by_coordinate': np.mean(diff**2, axis=0),
        'within_MSE_by_coordinate': np.mean(cdiff**2, axis=0),
        'zero_change_MSE_by_coordinate': np.mean(ac**2, axis=0),
        'predicted_change_MSE_by_coordinate': np.mean(pc**2, axis=0),
        'response_inner_product_by_coordinate': np.mean(pc*ac, axis=0)}


def denominators(actual, groups, cohorts):
    result = {}
    for cohort in sorted(set(cohorts)):
        take = np.asarray(cohorts) == cohort
        y = actual[take]
        result[cohort] = {'total': max(float(np.mean((y-y.mean(0))**2)), 1e-12),
            'within': max(float(np.mean(group_center(y, np.asarray(groups)[take], np.ones(len(y)))**2)), 1e-12)}
    return result


def summarize(arrays, cohorts, train_denominators):
    scores = {}
    for cohort in sorted(set(cohorts)):
        take = np.asarray(cohorts) == cohort
        values = {name.removesuffix('_by_question'): float(value[take].mean())
                  for name, value in arrays.items() if name.endswith('_by_question')}
        den = train_denominators[cohort]
        values['normalized_selection_objective'] = .5*(values['absolute_MSE']/den['total']+values['within_MSE']/den['within'])
        scores[cohort] = values
    scores['equal_cohort'] = {name: float(np.mean([scores[c][name] for c in sorted(set(cohorts))]))
                              for name in next(iter(scores.values()))}
    return scores
