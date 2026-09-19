"""Resume an unchanged admitted experiment without regenerating protocol time."""
from rdc_construction_common import *
import phase2746_rdc_history_autonomous as task


def main():
    path=task.OUT/'autonomous/protocol.json';original=read(path)
    # The original entry point recomputes a timestamp before an immutable
    # comparison. Correct only that administrative value; every scientific
    # field, source hash, fit, route, cohort and precision remains compared.
    regular=task.immutable
    def preserved_timestamp(target,value):
        if Path(target)==path and path.exists():value={**value,'timestamp':original['timestamp']}
        return regular(target,value)
    task.immutable=preserved_timestamp
    task.CUDA_TASKS=task.CUDA_TASKS|{Path(__file__).name}
    save(task.OUT/'autonomous/resume_entry.json',{'timestamp':stamp(),'source':snapshot(__file__),
        'original_implementation_sha256':sha(task.__file__),'protocol_sha256':sha(path),
        'cause':'Main resume initially rejected only the freshly regenerated protocol timestamp, before any model load or main-row write.',
        'fix':'Reuse original protocol timestamp for exact immutable comparison. No scientific field or kernel replaced; existing successful eight-row pilot remains intact.'})
    task.main(False)


if __name__=='__main__':main()
