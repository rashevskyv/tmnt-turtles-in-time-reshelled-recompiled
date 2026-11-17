#include <cstdlib>

#include <rex/logging.h>
#include <rex/system/kernel_state.h>
#include <rex/ui/window_nx.h>

extern "C" [[noreturn]] void rex_nx_quick_exit(const char* why, bool to_menu);  // switch/host/switch_nx.cpp
extern "C" const char* rex_nx_exit_request(void);  // switch_nx.cpp: an exit asked on another thread

namespace rex::ui {

NXWindowedAppContext::NXWindowedAppContext() { ueventCreate(&wake_event_, true); }

NXWindowedAppContext::~NXWindowedAppContext() {}

bool NXWindowedAppContext::Initialize() { return true; }

void NXWindowedAppContext::NotifyUILoopOfPendingFunctions() { ueventSignal(&wake_event_); }

void NXWindowedAppContext::PlatformQuitFromUIThread() { ueventSignal(&wake_event_); }

int NXWindowedAppContext::RunMainMessageLoop() {
  while (!HasQuitFromUIThread()) {
    // HOME -> Close and the game's own quit end the process at once (rex_nx_quick_exit, switch_nx.cpp): the SDK's
    // teardown raced the GPU command thread (hcu-switch 0.246). The game's quit goes back to hbmenu when it started
    // the NRO, HOME -> Close to the HOME menu.
    if (!appletMainLoop()) rex_nx_quick_exit("the system asked the application to exit (HOME -> Close)", false);
    if (auto* ks = REX_KERNEL_STATE(); ks && ks->is_terminating_title()) rex_nx_quick_exit("the game ended the title", true);
    if (const char* why = rex_nx_exit_request()) rex_nx_quick_exit(why, true);
    // Wake on a request (paint, deferred function, quit) or every 16 ms for the applet pump.
    waitSingle(waiterForUEvent(&wake_event_), 16'000'000ULL);
    ExecutePendingFunctionsFromUIThread();
    if (HasQuitFromUIThread()) {
      break;
    }
    if (window_ && window_->TakePaintPending()) {
      window_->HandlePaintEvent();
    }
  }
  return EXIT_SUCCESS;
}

}  // namespace rex::ui
