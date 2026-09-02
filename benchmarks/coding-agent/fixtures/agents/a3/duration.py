import re


def parse_duration(value: str) -> int:
    """Parse a compact duration and return milliseconds.

    Accepted units: ms, s, m. Whitespace around the value is allowed.
    """
    text=value.strip().lower()
    m=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(ms|s|m)',text)
    if not m:
        raise ValueError(f'invalid duration: {value!r}')
    amount=float(m.group(1)); unit=m.group(2)
    if unit.endswith('s'):
        return int(amount*1000)
    if unit == 'ms':
        return int(amount)
    return int(amount*60_000)
