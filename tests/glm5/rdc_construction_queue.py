"""Finite, serial and resumable native-CUDA work list for Phase2745."""
import argparse, subprocess, sys
from rdc_construction_common import *


JOBS = [
    ('q14_scheduler', 'phase2745_rdc_construction_pilot.py', ['qwen14']),
    ('q14_source_interleaving', 'phase2745_rdc_construction_batch_pilot.py', ['qwen14']),
    ('q14_capture', 'phase2745_rdc_construction_capture.py', ['qwen14']),
    ('q4_scheduler', 'phase2745_rdc_construction_pilot.py', ['qwen4']),
    ('q4_source_interleaving', 'phase2745_rdc_construction_batch_pilot.py', ['qwen4']),
    ('q4_capture', 'phase2745_rdc_construction_capture.py', ['qwen4']),
    ('norm_pilot', 'phase2745_rdc_construction_norm_pilot.py', []),
    ('norm_directions', 'phase2745_rdc_construction_norm.py', []),
    ('glm_scheduler', 'phase2745_rdc_construction_pilot.py', ['glm4']),
    ('glm_source_interleaving', 'phase2745_rdc_construction_batch_pilot.py', ['glm4']),
    ('glm_capture', 'phase2745_rdc_construction_capture.py', ['glm4']),
    ('fit_preflight', 'phase2745_rdc_construction_fit_preflight.py', []),
    ('q4_full_operator', 'phase2745_rdc_construction_fit.py', ['qwen4']),
    ('q14_full_operator', 'phase2745_rdc_construction_fit.py', ['qwen14']),
    ('glm_full_operator', 'phase2745_rdc_construction_fit.py', ['glm4']),
    ('q4_diagonal', 'phase2745_rdc_construction_diagonal.py', ['qwen4']),
    ('q14_diagonal', 'phase2745_rdc_construction_diagonal.py', ['qwen14']),
    ('glm_diagonal', 'phase2745_rdc_construction_diagonal.py', ['glm4']),
    ('glm_interactions', 'phase2745_rdc_construction_analysis.py', ['glm4']),
    ('norm_analysis', 'phase2745_rdc_construction_norm_analysis.py', []),
    ('q4_native_compilation', 'phase2745_rdc_construction_compile.py', ['qwen4']),
    ('q14_native_compilation', 'phase2745_rdc_construction_compile.py', ['qwen14']),
    ('glm_native_compilation', 'phase2745_rdc_construction_compile.py', ['glm4']),
    ('q4_native_language_reuse', 'phase2745_rdc_construction_language.py', ['qwen4']),
    ('glm_native_language', 'phase2745_rdc_construction_language.py', ['glm4']),
    ('q14_native_language', 'phase2745_rdc_construction_language.py', ['qwen14']),
    ('scientific_figures', 'phase2745_rdc_construction_figures.py', [])]


def main(after_pid=None):
    import psutil
    out = BASE / 'queue'
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if after_pid is not None and psutil.pid_exists(after_pid):
        process = psutil.Process(after_pid)
        observed = ' '.join(process.cmdline())
        assert 'phase2745_rdc_construction_pilot.py' in observed and 'qwen14' in observed
        save(out / 'waiting_for_existing_pilot.json', {'timestamp': stamp(), 'pid': after_pid,
            'observed_command': observed, 'action': 'Read-only wait for this already running authorized pilot to exit; never stop or alter it.'})
        while process.is_running():
            guard()
            save(out / 'status.json', {'timestamp': stamp(), 'state': 'waiting_existing_pilot', 'pid': after_pid})
            try:
                process.wait(timeout=15)
            except psutil.TimeoutExpired:
                continue
            break
    receipts = []
    for number, (name, script, args) in enumerate(JOBS):
        guard()
        path = Path(__file__).with_name(script)
        assert path.is_file()
        # A manually launched CPU analysis can finish just after the queue
        # reaches it. Wait for that same script/model before the idempotent
        # entry point, rather than letting two writers race at its commit.
        own_chain = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for existing in psutil.process_iter(['pid', 'cmdline']):
            argv = existing.info['cmdline'] or []
            matches = [i for i, arg in enumerate(argv) if Path(arg).name == script]
            if existing.info['pid'] in own_chain or not any(argv[i+1:] == args for i in matches):
                continue
            while existing.is_running():
                save(out / 'status.json', {'timestamp': stamp(), 'state': 'waiting_same_registered_job',
                    'job': name, 'pid': existing.pid, 'command': argv, 'goal_complete': False})
                try:
                    existing.wait(timeout=15)
                except psutil.TimeoutExpired:
                    guard()
                    continue
                break
        source = snapshot(path)
        command = [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-X', 'faulthandler', str(path), *args]
        runid = str(time.time_ns())
        log = out / (name+'_'+runid+'.log')
        tick = time.monotonic()
        print('CONSTRUCTION_QUEUE_START', number+1, len(JOBS), name, flush=True)
        with log.open('wb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            while child.poll() is None:
                save(out / 'status.json', {'timestamp': stamp(), 'state': 'running', 'job': name,
                     'position': number+1, 'total': len(JOBS), 'pid': child.pid,
                     'source': source, 'log': str(log.relative_to(BASE)), 'command': command,
                     'completed_in_this_invocation': receipts, 'goal_complete': False})
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    continue
        record = {'timestamp': stamp(), 'job': name, 'source': source, 'command': command,
            'exit_code': child.returncode, 'seconds': time.monotonic()-tick, 'log': str(log.relative_to(BASE))}
        save(out / 'receipts' / (name+'_'+runid+'.json'), record)
        receipts.append(record)
        print('CONSTRUCTION_QUEUE_EXIT', name, child.returncode, round(record['seconds'], 1), flush=True)
        if child.returncode:
            save(out / 'status.json', {'timestamp': stamp(), 'state': 'needs_diagnostic', 'failed_job': name,
                                     'receipts': receipts, 'goal_complete': False})
            raise RuntimeError('Child failed; preserve artifacts and diagnose before resuming: '+name)
    save(out / 'status.json', {'timestamp': stamp(), 'state': 'listed_jobs_complete', 'receipts': receipts,
        'seconds': time.monotonic()-started, 'goal_complete': False,
        'remaining': 'Native output compilation, scientific summaries, complete client/MEMO delivery, then the integrated same-goal phases.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int)
    main(parser.parse_args().after_pid)
