<div>
    <label>
        Search
        <input type="search" wire:model.live.debounce.300ms="search">
    </label>

    <x-actions.button wire:click="resetFilters" wire:loading.attr="loading" wire:target="resetFilters">
        Reset
    </x-actions.button>

    <div data-project-results>
        {{-- project rows are rendered here --}}
    </div>
</div>
