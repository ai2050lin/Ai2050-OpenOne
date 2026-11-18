"""Keep the frozen scientific runner; check the actual relocated result volume."""
from pathlib import Path
import shutil
import sys
import mlp_trigger_design as D
import mlp_trigger_run as R
import mlp_trigger_read100 as S

def main():
    S.configure();p=D.checked();actual=D.OUT.resolve();assert actual.drive.upper()=='C:'
    original=shutil.disk_usage
    def actual_volume(path):
        return original(actual if Path(path)==D.ROOT else path)
    D.save('storage-execution-amendment.json',{'schema':'resource_path_fix.v1','adapter_sha256':D.HASH(__file__),
       'logical_root':str(D.OUT),'actual_root':str(actual),'original_guard':'disk_usage(project_root)',
       'corrected_guard':'same byte threshold on actual result-storage volume',
       'unchanged':'scientific runner bytes, inputs, model, doses, predictions, thresholds and saved result schema',
       'source_unchanged_verified':True},p)
    shutil.disk_usage=actual_volume
    R.main()
if __name__=='__main__':main()
