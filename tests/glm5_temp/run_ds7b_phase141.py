"""DS7B Phase 141 运行脚本 - 写入文件"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

LOG_FILE = 'tests/glm5_temp/phase141_ds7b_log2.txt'

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()
    print(msg, flush=True)

log(f"Starting DS7B Phase 141")
log(f"PyTorch: {__import__('torch').__version__}")

sys.path.insert(0, 'tests/glm5')

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from tests.glm5.phase141_jacobian_manifold import run_phase141
    log("Module imported successfully")
    
    log("Running Phase 141 for DS7B...")
    results = run_phase141('deepseek7b')
    
    # 保存结果
    import json
    timestamp = time.strftime("%Y%m%d_%H%M")
    filename = f"tests/glm5_temp/phase141_deepseek7b_jacobian_manifold_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {filename}")
    log("DS7B Phase 141 COMPLETE!")
    
except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
