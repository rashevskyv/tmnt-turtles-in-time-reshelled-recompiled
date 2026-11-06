#!/usr/bin/env python3
"""GCC/Horizon fixes for this repository's patches/rexglue-sdk, run by switch/build.sh after the shared
switch/sdk-edits.py (cvhd-recomp's SDK has none of these patches). The same fixes as hcu-switch's
sdk-edits-hcu.py: Hard Corps: Uprising Recompiled has the same patch baseline.

    python3 switch/sdk-edits-tmnt.py /root/tmntu/rexglue-sdk

Idempotent: an edit whose new text is already there is skipped.
"""
import os
import sys

SDK = sys.argv[1]
NL = chr(10)
changed = []


def edit(rel, pairs):
    path = os.path.join(SDK, rel)
    s = open(path, encoding='utf-8').read()
    orig = s
    for old, new in pairs:
        if new in s:
            continue
        if old not in s:
            raise SystemExit(f'{rel}: anchor not found:\n{old}')
        s = s.replace(old, new, 1)
    if s != orig:
        open(path, 'w', encoding='utf-8', newline=NL).write(s)
        changed.append(rel)


# 0009 guest_wake.cpp: REX_PLATFORM_LINUX is also set on the Switch, which has no <linux/futex.h>.
# Nothing on the Switch blocks on a guest word, so the wake stays a no-op there.
edit('src/graphics/guest_wake.cpp', [
    ('#elif REX_PLATFORM_LINUX' + NL + '#include <climits>' + NL,
     '#elif REX_PLATFORM_LINUX && !REX_PLATFORM_SWITCH' + NL + '#include <climits>' + NL),
    ('#elif REX_PLATFORM_LINUX' + NL + '  std::atomic_thread_fence',
     '#elif REX_PLATFORM_LINUX && !REX_PLATFORM_SWITCH' + NL + '  std::atomic_thread_fence'),
])

# 0006 RtlEnterCriticalSection spin: __builtin_arm_yield is clang's; GCC (devkitA64) takes the instruction.
edit('src/kernel/xboxkrnl/xboxkrnl_rtl.cpp', [
    ('#elif defined(__aarch64__)' + NL + '  __builtin_arm_yield();' + NL,
     '#elif defined(__aarch64__)' + NL + '  __asm__ volatile("yield");' + NL),
])

# 0045 ImGui mouse cursors through SDL: the Switch has no SDL and no mouse cursor.
edit('src/ui/imgui_drawer.cpp', [
    ('#include <SDL3/SDL.h>' + NL,
     '#if !REX_PLATFORM_SWITCH' + NL + '#include <SDL3/SDL.h>' + NL + '#endif' + NL),
    ('void ApplyMouseCursor() {' + NL,
     'void ApplyMouseCursor() {' + NL + '#if REX_PLATFORM_SWITCH' + NL + '  return;' + NL + '#else' + NL),
    ('    SDL_SetCursor(cursors[cursor]);' + NL + '    applied = cursor;' + NL + '  }' + NL + '}' + NL,
     '    SDL_SetCursor(cursors[cursor]);' + NL + '    applied = cursor;' + NL + '  }' + NL + '#endif' + NL + '}' + NL),
])

# 0071 audio callback pacing sleeps with SDL_DelayNS: no SDL on the Switch, the same sleep through libstdc++.
edit('src/audio/audio_system.cpp', [
    ('#include <SDL3/SDL_timer.h>' + NL,
     '#if REX_PLATFORM_SWITCH' + NL + '#include <thread>' + NL +
     'static inline void SDL_DelayNS(uint64_t ns) { std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); }' + NL +
     '#else' + NL + '#include <SDL3/SDL_timer.h>' + NL + '#endif' + NL),
])

print('changed (tmnt):', *changed, sep=NL + '  ')
