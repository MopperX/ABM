@props(['value'=>null,'type'=>'normal','variant'=>'default','checked'=>false,'slotName'=>null])
<wa-dropdown-item value="{{ $value }}" type="{{ $type }}" variant="{{ $variant }}" @if($checked) checked @endif @if($slotName) slot="{{ $slotName }}" @endif>{{ $slot }}</wa-dropdown-item>
