from pathlib import Path
import re, subprocess, sys
php=Path('app/Livewire/ProjectList.php').read_text()
blade=Path('resources/views/livewire/project-list.blade.php').read_text()
docs=Path('docs/custom-webawesome-components.md').read_text()

def must(cond,msg):
    if not cond: raise AssertionError(msg)
# Livewire state + action
must(re.search(r'public\s+string\s+\$status\s*=\s*[\'\"]all[\'\"]',php) is not None,'status property defaults to all')
must(re.search(r'function\s+selectStatus\s*\(\s*string\s+\$status\s*\)',php) is not None,'selectStatus method')
must("['all', 'active', 'paused', 'archived']" in php or '["all", "active", "paused", "archived"]' in php,'allowed statuses')
must(re.search(r"in_array\s*\(\s*\$status",php) is not None,'validate status')
must(re.search(r'\$this->status\s*=\s*\$status',php) is not None,'assign status')
must(re.search(r"\$this->status\s*=\s*['\"]all['\"]",php) is not None,'reset status')
# Project wrappers + documented event flow
must('<x-actions.dropdown' in blade,'project dropdown wrapper')
must('<x-actions.button' in blade,'project button wrapper')
must('slot-name="trigger"' in blade or "slot-name='trigger'" in blade,'trigger uses slot-name')
must('x-on:wa-select=' in blade,'wa-select event')
must('$wire.selectStatus($event.detail.item.value)' in blade,'selection calls Livewire')
for val in ['all','active','paused','archived']:
    must(re.search(rf'<x-actions\.dropdown-item[^>]*value=[\'\"]{val}[\'\"]',blade) is not None,f'{val} item')
must('<wa-dropdown' not in blade and '<wa-button' not in blade,'no raw WA elements')
must(re.search(r'<x-actions\.button-group[^>]*label=',blade) is not None,'accessible button group label')
must('wire:loading.attr="loading"' in blade and 'wire:target="resetFilters"' in blade,'scoped loading state')
# PHP syntax when available
cp=subprocess.run(['php','-l','app/Livewire/ProjectList.php'],capture_output=True,text=True)
must(cp.returncode==0,'PHP syntax: '+cp.stderr)
print('18 passed')
