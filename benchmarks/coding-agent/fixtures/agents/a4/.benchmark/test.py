import sys
sys.path.insert(0, '.')
from src.routing import normalize_route
cases={
 'api/v1/Users':'/api/v1/Users',
 ' /API//V2/Items/ ':'/API/V2/Items',
 '/':'/',
 '///Health':'/Health',
 '/Mixed/Case/':'/Mixed/Case',
}
for v,e in cases.items():
    got=normalize_route(v); assert got==e,(v,got,e)
print('5 passed')
