"""Bounded CPU reconstruction of old source-RMS and prefix-only query proposal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'glm5'))
from threadpoolctl import threadpool_limits
from rdc_joint_prior_rules import PriorRMS, QueryProposal

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        old = PriorRMS()
        print('PRIOR_RMS_HASH_RECONSTRUCTION_PASSED', flush=True)
        query = QueryProposal()
        print('QUERY_PROPOSAL_FROZEN_BEFORE_NEW_MAIN_CAPTURE', flush=True)
