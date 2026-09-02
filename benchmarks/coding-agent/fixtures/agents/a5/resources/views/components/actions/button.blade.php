@props(['route' => null, 'routeParameters' => [], 'disabled' => false, 'slotName' => null])
@php
$attrs=[];
if ($route) { $attrs['href']=route($route,$routeParameters); $attrs['wire:navigate']=true; }
if ($disabled) $attrs['disabled']=true;
if ($slotName) $attrs['slot']=$slotName;
@endphp
<wa-button {{ $attributes->merge($attrs) }}>{{ $slot }}</wa-button>
