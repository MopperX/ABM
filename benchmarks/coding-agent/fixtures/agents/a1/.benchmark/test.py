import importlib.util
from pathlib import Path
p=Path('stats.py')
spec=importlib.util.spec_from_file_location('stats',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def check(cond,msg):
    if not cond: raise AssertionError(msg)
check(m.median([3]) == 3, 'single')
check(m.median([3,1,2]) == 2, 'odd')
check(m.median([1,2,3,4]) == 2.5, 'even')
check(m.median([4,1,3,2]) == 2.5, 'even unsorted')
x=[9,1,5,3]; before=x.copy(); m.median(x); check(x==before,'must not mutate')
try: m.median([])
except ValueError: pass
else: raise AssertionError('empty must raise ValueError')
print('6 passed')
