def normalize_route(path: str) -> str:
    """Normalize an application route."""
    path = path.strip()
    if not path.startswith('/'):
        path = '/' + path
    while '//' in path:
        path = path.replace('//', '/')
    if len(path) > 1:
        path = path.rstrip('/')
    return path.lower()
