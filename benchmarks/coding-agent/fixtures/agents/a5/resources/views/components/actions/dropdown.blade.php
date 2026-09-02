@props(['open' => false, 'placement' => 'bottom'])
<wa-dropdown {{ $attributes->merge(['placement'=>$placement, 'open'=>$open ?: null]) }}>{{ $slot }}</wa-dropdown>
