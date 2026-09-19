"""Entry point for stored exact native fields -> frozen available feature blocks."""
from rdc_joint_common import *
from phase2719_rdc_archive_recovery import load_archived
from rdc_joint_features import build


def main(fresh=False):
    if fresh:assert (BASE/'frozen.json').exists()
    store = load_archived(fresh)
    try:build(store)
    finally:store.clear()


if __name__=='__main__':
    import argparse
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--fresh',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):main(a.fresh)
