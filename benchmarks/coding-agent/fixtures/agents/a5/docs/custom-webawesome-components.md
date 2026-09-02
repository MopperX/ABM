# Project UI conventions

Use the project's Blade wrappers rather than raw Web Awesome elements.

## Button
Use `<x-actions.button>`. For Laravel navigation prefer the `route` prop. A dropdown trigger uses `slot-name="trigger"` because `slot` is reserved by Blade. With slotted end/start decorations, include the matching SSR hint such as `with-caret`, `with-start`, or `with-end`.

For Livewire actions, put `wire:click` on the button and scope busy feedback with `wire:loading.attr="loading"` and `wire:target`.

## Dropdown
Use `<x-actions.dropdown>` and `<x-actions.dropdown-item>`. Selection is emitted as `wa-select`; in Livewire use Alpine, for example `x-on:wa-select="$wire.selectStatus($event.detail.item.value)"`.

Give actionable dropdown items a stable `value`. Boolean Blade props must be bound (`:checked="$value"`) when dynamic.

## Button group
Use `<x-actions.button-group label="...">` for semantically related controls. The label is required for accessibility. Put Livewire actions on individual buttons, not on the group.
