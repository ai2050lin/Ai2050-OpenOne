"""Exact receipt after actual six revised screenshot inspections."""
from rdc_formation_common import *


def main():
    current=read(OUT/'client/current_evidence_regression.json');folder=ROOT/current['image_directory']
    result=read(ROOT/current['result']);assert result['all_passed'] and sha(ROOT/current['result'])==current['result_sha256']
    images=[{'path':str((folder/name).relative_to(BASE)),'sha256':sha(folder/name)} for name in result['images']]
    assert len(images)==6
    immutable(folder/'visual_review.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'images':images,'reviewer':'Main agent after actually viewing all six revised screenshots.',
        'checked':['All20layer H/gate and all96step postnorm/37H axes remain legible with exact native widths.',
            'Scientific panel has visible precision/no-change warnings; axes and caption are not clipped.',
            'Revised mobile long run names wrap inside panel and native controls fit without overflow.',
            'First mobile revision is retained as a failed visual example; no scientific array or scoring change.']})
    print('FORMATION_EVIDENCE_VISUAL_REVIEW',len(images),flush=True)


if __name__=='__main__':main()
