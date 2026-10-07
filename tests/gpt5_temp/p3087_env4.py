# -*- coding: utf-8 -*-
"""p3087 env probe 4: venv packages relevant to loading."""
import io

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3087_env4.txt'
o = []
for pkg in ('bitsandbytes', 'accelerate',
            'transformers', 'torch', 'numpy',
            'safetensors'):
    try:
        m = __import__(pkg)
        o.append('%s = %s'
                 % (pkg,
                    getattr(m, '__version__',
                            '??')))
    except Exception as e:
        o.append('%s MISSING (%s)'
                 % (pkg, type(e).__name__))
io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
