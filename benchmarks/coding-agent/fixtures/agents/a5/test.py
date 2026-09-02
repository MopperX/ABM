from pathlib import Path
import re


def test_status_state_and_project_dropdown():
    php = Path("app/Livewire/ProjectList.php").read_text()
    blade = Path("resources/views/livewire/project-list.blade.php").read_text()
    assert re.search(r"public\s+string\s+\$status\s*=\s*['\"]all['\"]", php)
    assert "<x-actions.dropdown" in blade
    assert "$wire.selectStatus($event.detail.item.value)" in blade


if __name__ == "__main__":
    test_status_state_and_project_dropdown()
    print("1 public test passed")
