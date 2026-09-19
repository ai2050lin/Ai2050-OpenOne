"""Full-coordinate ridge with an explicit within-context response objective.

This is a known weighted least-squares construction, NOT a new language law.
Contrasts modify the fitting loss only. Predictions are direct coordinate
vectors, never a response difference transplanted between model histories.
No native model execution or dataset access occurs in this module.
"""
import numpy as np


def group_center(values, groups, weights):
    """Subtract the weighted mean of each declared context group."""
    values=np.asarray(values,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    groups=np.asarray(groups)
    assert values.shape[0]==len(weights)==len(groups)
    assert np.isfinite(values).all() and np.isfinite(weights).all() and (weights>0).all()
    result=values.copy()
    for group in np.unique(groups):
        selected=groups==group
        mean=np.tensordot(weights[selected]/weights[selected].sum(),values[selected],axes=(0,0))
        result[selected]-=mean
    return result


def contrast_transform(values, groups, weights, strength):
    """B X with B^T B = W + strength * P^T W P.

    W=diag(weights), P subtracts weighted context means. With U's orthonormal
    columns sqrt(w_i/sum_context_w), B=[I+(sqrt(1+a)-1)(I-UU^T)] sqrt(W).
    All rows and coordinates are retained; no low-rank spectral truncation.
    """
    values=np.asarray(values,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    assert np.isfinite(strength) and strength>=0
    scale=weights.reshape((-1,)+(1,)*(values.ndim-1))**.5
    return scale*(values+(np.sqrt(1+strength)-1)*group_center(values,groups,weights))


def objective(errors, groups, weights, strength):
    errors=np.asarray(errors,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    centered=group_center(errors,groups,weights)
    return float(np.einsum('n,nd,nd->',weights,errors,errors)
        +strength*np.einsum('n,nd,nd->',weights,centered,centered))


class SelectivityRidge:
    """Ordinary/direct-output ridge and its within-context-weighted variants.

    Input values must be acquired from the permitted early prefix boundary;
    this class cannot certify their causal availability. All normalization is
    fit on the declared training rows. The intercept remains unpenalized.
    """
    def __init__(self, strength=0., ridge=1., standardize=True):
        assert np.isfinite(strength) and strength>=0
        assert np.isfinite(ridge) and ridge>0
        self.strength=float(strength);self.ridge=float(ridge);self.standardize=bool(standardize)

    def fit(self, features, targets, groups, weights):
        x=np.asarray(features,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
        w=np.asarray(weights,dtype=np.float64)
        assert x.ndim==y.ndim==2 and len(x)==len(y)==len(w)==len(groups)
        assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all() and (w>0).all()
        w=w/w.sum();self.x_mean=np.einsum('n,nd->d',w,x);self.y_mean=np.einsum('n,nd->d',w,y)
        xx=x-self.x_mean;yy=y-self.y_mean
        self.x_scale=np.sqrt(np.einsum('n,nd,nd->d',w,xx,xx)) if self.standardize else np.ones(x.shape[1])
        self.x_scale=np.maximum(self.x_scale,1e-8);xx=xx/self.x_scale
        bx=contrast_transform(xx,groups,w,self.strength);by=contrast_transform(yy,groups,w,self.strength)
        gram=bx@bx.T;gram=(gram+gram.T)/2
        eigenvectors=None
        eigenvalues,eigenvectors=np.linalg.eigh(gram)
        tolerance=1e-10*max(1.,float(np.abs(eigenvalues).max()))
        assert eigenvalues.min()>=-tolerance,('Unexpected non-PSD Gram',float(eigenvalues.min()),tolerance)
        # Only tiny negative roundoff is clipped. Every eigencomponent is used.
        eigenvalues=np.maximum(eigenvalues,0.)
        solved=eigenvectors@((eigenvectors.T@by)/(eigenvalues[:,None]+self.ridge))
        self.coefficients=bx.T@solved
        self.audit={'training_rows':len(x),'feature_coordinates':x.shape[1],'target_coordinates':y.shape[1],
            'context_groups':len(np.unique(groups)),'strength':self.strength,'ridge':self.ridge,
            'standardize_train_only':self.standardize,
            'effective_df':float(np.sum(eigenvalues/(eigenvalues+self.ridge))),
            'eigenvalues_retained':len(eigenvalues),'top_k_or_PCA':False,
            'maximum_coefficient_magnitude':float(np.abs(self.coefficients).max()),
            'scope':'Direct full-coordinate predictor, with a known weighted-loss contrast term. '
                    'Only declared available input features are supplied by predict(); no target hidden state, '
                    'answer label, future token or target context-group mean is an input.'}
        return self

    def predict(self, features):
        x=np.asarray(features,dtype=np.float64)
        assert x.ndim==2 and x.shape[1]==len(self.x_mean) and np.isfinite(x).all()
        return (x-self.x_mean)/self.x_scale@self.coefficients+self.y_mean


def contrast_scores(predicted, actual, groups, weights):
    """Group-centered scores are retrospective evaluation, never model input."""
    predicted=np.asarray(predicted,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64);weights=weights/weights.sum()
    assert predicted.shape==actual.shape and predicted.ndim==2
    diff=predicted-actual;pc=group_center(predicted,groups,weights);ac=group_center(actual,groups,weights)
    def square(v):return float(np.einsum('n,nd,nd->',weights,v,v)/v.shape[1])
    return {'absolute_full_coordinate_MSE':square(diff),'within_context_response_MSE':square(pc-ac),
        'zero_response_change_MSE':square(ac),'predicted_response_amplitude':square(pc),
        'within_context_response_inner_product':float(np.einsum('n,nd,nd->',weights,pc,ac)/actual.shape[1]),
        'scope':'Actual responses used only to score predictions. Zero-change is a contrast baseline, '
                'not an available absolute target vector or a claim about native semantic invariance.'}


class SelectivityKernel:
    """Exact full-coordinate additive/bilinear features without materializing D^2.

    k((q,c),(q',c')) = q.q'/Dq + c.c'/Dc
    + interaction*(q.q'/Dq)*(c.c'/Dc).

    The product term contains EVERY query-coordinate/context-coordinate pair;
    it is not a Top-K list, PCA projection, or a neural attention algorithm.
    All outputs remain in their original target coordinate order. Kernel and
    feature centering, and the unpenalized intercept, use only training rows.
    """
    def __init__(self, strength=0., ridge=1., interaction=0.):
        assert np.isfinite(strength) and strength >= 0
        assert np.isfinite(ridge) and ridge > 0
        assert np.isfinite(interaction) and interaction >= 0
        self.strength = float(strength)
        self.ridge = float(ridge)
        self.interaction = float(interaction)

    @staticmethod
    def _normalize_train(values, weights):
        values = np.asarray(values, dtype=np.float64)
        assert values.ndim == 2 and values.shape[1] and np.isfinite(values).all()
        mean = weights @ values
        scale = np.maximum(np.sqrt(weights @ ((values-mean)**2)), 1e-8)
        return (values-mean)/scale, mean, scale

    def _kernel(self, query, context):
        kq = query @ self.train_query.T / query.shape[1]
        kc = context @ self.train_context.T / context.shape[1]
        return kq + kc + self.interaction*kq*kc

    def fit(self, query, context, targets, groups, weights):
        query = np.asarray(query, dtype=np.float64)
        context = np.asarray(context, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        assert targets.ndim == 2 and len(query) == len(context) == len(targets) == len(groups) == len(weights)
        assert np.isfinite(targets).all() and np.isfinite(weights).all() and (weights > 0).all()
        self.weights = weights/weights.sum()
        self.train_query, self.query_mean, self.query_scale = self._normalize_train(query, self.weights)
        self.train_context, self.context_mean, self.context_scale = self._normalize_train(context, self.weights)
        self.target_mean = self.weights @ targets
        raw = self._kernel(self.train_query, self.train_context)
        self.kernel_column_mean = self.weights @ raw
        self.kernel_grand_mean = float(self.kernel_column_mean @ self.weights)
        centered = raw - (raw @ self.weights)[:, None] - self.kernel_column_mean[None, :] + self.kernel_grand_mean
        n = len(weights)
        b = contrast_transform(np.eye(n), groups, self.weights, self.strength)
        gram = b @ centered @ b.T
        gram = (gram+gram.T)/2
        ev, vectors = np.linalg.eigh(gram)
        tolerance = 1e-10*max(1., float(np.abs(ev).max()))
        assert ev.min() >= -tolerance, ('Unexpected non-PSD kernel', float(ev.min()), tolerance)
        ev = np.maximum(ev, 0.)
        rhs = b @ (targets-self.target_mean)
        dual = vectors @ ((vectors.T @ rhs)/(ev[:, None]+self.ridge))
        self.coefficients = b.T @ dual
        self.audit = {
            'training_rows': n, 'query_coordinates': query.shape[1], 'context_coordinates': context.shape[1],
            'target_coordinates': targets.shape[1], 'strength': self.strength,
            'ridge': self.ridge, 'interaction': self.interaction,
            'bilinear_coordinate_pairs': int(query.shape[1]*context.shape[1]) if self.interaction else 0,
            'eigenvalues_retained': n, 'effective_df': float(np.sum(ev/(ev+self.ridge))),
            'top_k_or_PCA': False, 'train_only_normalization_and_kernel_centering': True,
            'scope': 'Known exact polynomial feature kernel with direct original-coordinate targets. '
                     'No semantic causality, native-parameter identity, or complete mechanism is implied.',
        }
        return self

    def predict(self, query, context):
        query = np.asarray(query, dtype=np.float64)
        context = np.asarray(context, dtype=np.float64)
        assert query.ndim == context.ndim == 2 and len(query) == len(context)
        assert query.shape[1] == len(self.query_mean) and context.shape[1] == len(self.context_mean)
        assert np.isfinite(query).all() and np.isfinite(context).all()
        raw = self._kernel((query-self.query_mean)/self.query_scale,
                           (context-self.context_mean)/self.context_scale)
        centered = raw - (raw @ self.weights)[:, None] - self.kernel_column_mean[None, :] + self.kernel_grand_mean
        return centered @ self.coefficients + self.target_mean

    def bilinear_fitted_coefficient(self, query_coordinate, context_coordinate):
        """Return a requested fitted coordinate-pair coefficient for ALL outputs.

        This is an extracted predictor coefficient, NOT a native LLM weight.
        Coordinates refer to train-standardized inputs and unscaled q_i*c_j.
        No source pair is selected by amplitude; any valid pair is addressable.
        """
        assert 0 <= query_coordinate < self.train_query.shape[1]
        assert 0 <= context_coordinate < self.train_context.shape[1]
        gamma = self.coefficients-self.weights[:, None]*self.coefficients.sum(axis=0, keepdims=True)
        products = self.train_query[:, query_coordinate]*self.train_context[:, context_coordinate]
        return self.interaction*(products @ gamma)/(self.train_query.shape[1]*self.train_context.shape[1])
