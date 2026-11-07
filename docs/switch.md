# Nintendo Switch (experimental)

A homebrew NRO for a Switch running Atmosphère, built from the same recompiled game code as the PC port.
**No game files are included**: copy the files of your own package to the SD card.

## On the console

| Path on the SD card | What it holds |
| --- | --- |
| `/switch/tmnt/tmnt.nro` | The program (start it from hbmenu, with full RAM: hold R while you open a game) |
| `/switch/tmnt/game/` | Your package's files: `default.xex`, `TMNT_r.xex`, `Data/`, ... (the same files as the PC `game` folder) |
| `/switch/tmnt/deko3d`, `/switch/tmnt/shaders/` | The renderer marker and its compiled shaders |
| `/switch/tmnt/tmnt_log.txt` | The log of the last run |

The licence is the full game (`--license_mask`, as `config/tmnt_turtles_in_time_reshelled.toml` on PC).
Rumble works. **L3 + R3** opens the settings menu: frame rate, rumble, volume, fps counter, button icons,
skip intro and the language (en, fr, de, it, es, pt-BR; the last two after a restart).
The game hooks of the PC build are in, except online play: the FIRE menu has no Xbox Live entries.

### GPU fault capture

Create an empty `/switch/tmnt/dk_gpu_checkpoints` file on the SD card and restart the game.
Save any previous `dk_gpu_fault.txt` and `dk_gpu_pending.txt` before testing.
Reproduce the fault, then collect the new reports and `tmnt_log.txt` from the same folder.
The report shows the last completed checkpoint and the next checkpoint in the command list.
If the fence stays busy for 120 skipped frames, `dk_gpu_pending.txt` records a snapshot while the GPU may still be running.
With the marker present, the renderer ignores the `dk_skip` shader list.
The extra GPU barriers can change when the fault occurs. Remove the marker to return to normal timing.

## Building

The build runs in WSL or Linux with devkitPro (devkitA64, libnx, deko3d, uam) and a mesa-switch NVK build.

1. ReXGlue v0.10.0 at the commit in `patches/rexglue-sdk/baseline.env`, with every common and `.linux` patch of
   `patches/rexglue-sdk` applied and committed (the build resets the SDK to that commit when its edits change).
   Build the host CLI from it (`cmake --preset linux-amd64`, target `rexglue`, clang 20).
2. Your package's files in `$TMNT_ROOT/src/game/extracted`.
3. `switch/build.sh` copies the checkout to `$TMNT_ROOT/src`, applies `switch/sdk-edits.py` and
   `switch/sdk-edits-tmnt.py` to the SDK, runs codegen and builds `tmnt.nro` plus `stage/` (the NRO, the
   shaders as DKSH). The hbmenu icon is made from your `banner.png` and `title_icon.png` (kept outside git).
4. `switch/deploy.sh push` copies the stage to the console over FTP; every file is read back and checked by SHA-1.

## How it is put together

| Part | Files |
| --- | --- |
| CMake | `CMakeLists.txt` hands a Switch build to `switch/switch.cmake` and returns; the PC targets are untouched |
| Entry | `switch/tmnt_nx_main.cpp`: the plain ReXApp with the game hooks, without the launcher, online play and overlays |
| Host layer | `switch/host/`: memory, fibers, exceptions, threads, window, input with rumble, the settings menu |
| SDK edits | `switch/sdk-edits.py` (Horizon support), `switch/sdk-edits-tmnt.py` (GCC fixes for this repo's patches) |
| Renderer | `src/gpu/deko3d_backend.cpp` (deko3d), `src/gpu/trace_backend.cpp` (shader dumps), `tools/xenos2glsl.py` |
| TMNT_r.xex | linked into the executable (Horizon has no `dlopen`), its function table from `tools/dlc_modules.py tables` |

The host layer, the SDK edits and the renderer come from the Castlevania: Harmony of Despair Switch port at the
commit in `switch/CVHD_COMMIT`.
