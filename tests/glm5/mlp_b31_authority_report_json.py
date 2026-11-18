"""Serialize the frozen analysis without changing any estimator or gate."""
from pathlib import Path

import numpy as np

import mlp_b31_authority_v2 as version
version.configure()

import mlp_b31_authority_design as D
import mlp_b31_authority_report as analysis


def builtin(value):
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {builtin(k):builtin(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [builtin(x) for x in value]
    return value


def save(path,data):
    data=builtin(data)
    data['serialization_repair']={
        'source_sha256':D.HASH(Path(__file__)),
        'frozen_analysis_sha256':D.HASH(Path(analysis.__file__)),
        'scope':'Convert NumPy scalar types to the identical builtin JSON values. No data, formula or gate changes.',
    }
    D.SAVE(path,data)


if __name__=='__main__':
    if not (D.OUT/'report.json').exists():
        analysis.SAVE=save
        analysis.report(D.checked())
