<?php
namespace App\Livewire;

class ProjectList
{
    public string $search = '';

    public function resetFilters(): void
    {
        $this->search = '';
    }
}
