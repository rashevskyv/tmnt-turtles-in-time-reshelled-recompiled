// Switch entry: the plain ReXApp, without the PC launcher, window host, online service and overlays
// (src/tmnt_turtles_in_time_reshelled_app.h). The cvars of config/tmnt_turtles_in_time_reshelled.toml that
// the game needs (the full licence) come from switch/host/ui/windowed_app_main_nx.cpp.
#include "generated/default/tmnt_turtles_in_time_reshelled_init.h"

#include <rex/rex_app.h>

class TmntNxApp : public rex::ReXApp {
 public:
  using rex::ReXApp::ReXApp;

  static std::unique_ptr<rex::ui::WindowedApp> Create(rex::ui::WindowedAppContext& ctx) {
    return std::unique_ptr<TmntNxApp>(new TmntNxApp(ctx, "tmnt_turtles_in_time_reshelled", PPCImageConfig));
  }
};

REX_DEFINE_APP(tmnt_turtles_in_time_reshelled, TmntNxApp::Create)
