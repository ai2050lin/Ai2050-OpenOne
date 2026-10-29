import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('BASE=', BASE)
print('cwd=', os.getcwd())
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'deploy'))
print('p0=', sys.path[0]); print('p1=', sys.path[1])
print('svc exists=', os.path.isfile(os.path.join(BASE, 'deploy', 'distributed_service.py')))
try:
    import distributed_service
    print('flat import OK, ADMIN len', len(distributed_service.ADMIN_TOKEN))
except Exception as e:
    print('flat import FAIL:', e)
try:
    import deploy.distributed_service as d2
    print('pkg import OK')
except Exception as e:
    print('pkg import FAIL:', type(e).__name__, e)
