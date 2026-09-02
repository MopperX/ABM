from dataclasses import dataclass

@dataclass(frozen=True)
class Project:
    id: int
    name: str
    status: str


def list_projects(projects, query=None):
    """Return projects matching the optional case-insensitive name query."""
    rows = list(projects)
    if query:
        needle = query.casefold()
        rows = [p for p in rows if needle in p.name.casefold()]
    return sorted(rows, key=lambda p: p.name.casefold())
