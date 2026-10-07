import py_compile, traceback
out = r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\compile3043.txt'
try:
    py_compile.compile(
        r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
        r'\phase3043_omega_p40_field_variance_qwen.py',
        doraise=True)
    msg = 'COMPILE OK'
except Exception:
    msg = 'COMPILE FAIL\n' + traceback.format_exc()
with open(out, 'w', encoding='utf-8') as f:
    f.write(msg)
print('done')
