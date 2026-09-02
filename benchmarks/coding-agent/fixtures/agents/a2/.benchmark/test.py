import sys
sys.path.insert(0, '.')
from projects import Project,list_projects
rows=[Project(1,'Zulu','archived'),Project(2,'Alpha','active'),Project(3,'Beta','paused'),Project(4,'Alpine','active')]
def names(x): return [p.name for p in x]
assert names(list_projects(rows)) == ['Alpha','Alpine','Beta','Zulu']
assert names(list_projects(rows,query='alp')) == ['Alpha','Alpine']
assert names(list_projects(rows,status='active')) == ['Alpha','Alpine']
assert names(list_projects(rows,status='paused')) == ['Beta']
assert names(list_projects(rows,query='a',status='active')) == ['Alpha','Alpine']
assert list_projects(rows,status='missing') == []
# input is not changed
before=rows.copy(); list_projects(rows,status='active'); assert rows==before
print('7 passed')
