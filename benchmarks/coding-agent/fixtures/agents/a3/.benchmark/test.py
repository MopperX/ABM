import sys
sys.path.insert(0, '.')
from duration import parse_duration
cases={'250ms':250,'1s':1000,'1.5s':1500,'2m':120000,' 3 m ':180000,'0ms':0}
for v,e in cases.items():
    got=parse_duration(v); assert got==e,(v,got,e)
for v in ['', '1', 'ms', '-1s', '1h', '1.2.3s']:
    try: parse_duration(v)
    except ValueError: pass
    else: raise AssertionError(f'{v!r} should fail')
print('12 passed')
