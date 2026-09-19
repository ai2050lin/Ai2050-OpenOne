import sys, json
out = {}
out['python'] = sys.version
try:
    import torch
    out['torch'] = torch.__version__
    out['cuda'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out['gpu'] = torch.cuda.get_device_name(0)
except Exception as e:
    out['torch_err'] = str(e)
try:
    import transformers, safetensors, numpy
    out['transformers'] = transformers.__version__
    out['safetensors'] = safetensors.__version__
    out['numpy'] = numpy.__version__
except Exception as e:
    out['lib_err'] = str(e)
with open(r'D:\AI2050\Ai2050-OpenOne\tests\glm5_temp\env_probe_2873.json', 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out))
