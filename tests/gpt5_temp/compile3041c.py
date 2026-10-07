import py_compile, traceback
out = r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\compile3041c.txt'
try:
    py_compile.compile(
        r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\phase3041_closeout.py', doraise=True)
    msg = 'COMPILE OK'
except Exception:
    msg = 'COMPILE FAIL\n' + traceback.format_exc()
with open(out, 'w', encoding='utf-8') as f:
    f.write(msg)
print('done')
