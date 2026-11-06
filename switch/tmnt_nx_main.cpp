// Switch entry: the plain ReXApp, without the PC launcher, window host, online service and overlays
// (src/tmnt_turtles_in_time_reshelled_app.h). The cvars of config/tmnt_turtles_in_time_reshelled.toml that
// the game needs (the full licence) come from switch/host/ui/windowed_app_main_nx.cpp.
#include "generated/default/tmnt_turtles_in_time_reshelled_init.h"

#include <string>

#include <rex/cvar.h>
#include <rex/nx_menu.h>
#include <rex/rex_app.h>

#include "language.h"
#include "localization/active_language.h"
#include "title_bgm_level.h"

// The launcher's settings (src/settings.cpp) that the hooks read; on the Switch they come from the L3+R3 menu.
REXCVAR_DEFINE_BOOL(tmnt_skip_intro, false, "TMNT", "Start directly at the title screen");
REXCVAR_DEFINE_STRING(tmnt_language, "en", "TMNT", "Text language: en, fr, de, it, es or pt-BR");

class TmntNxApp : public rex::ReXApp {
 public:
  using rex::ReXApp::ReXApp;

  static std::unique_ptr<rex::ui::WindowedApp> Create(rex::ui::WindowedAppContext& ctx) {
    return std::unique_ptr<TmntNxApp>(new TmntNxApp(ctx, "tmnt_turtles_in_time_reshelled", PPCImageConfig));
  }

  std::optional<rex::PathConfig> OnFinalizePaths(const rex::PathConfig& defaults,
                                                 std::function<void(rex::PathConfig)> resume) override {
    rex::cvar::SetFlagByName("tmnt_skip_intro", rex_nx_setting_skip_intro() ? "true" : "false");
    // As tmnt::ApplyLanguage (src/settings.cpp); the menu's order is tmnt::kLanguages.
    const tmnt::LanguageSetting& language = tmnt::kLanguages[rex_nx_setting_language()];
    rex::cvar::SetFlagByName(tmnt::kLanguageSetting, language.key);
    rex::cvar::SetFlagByName("user_language", std::to_string(language.xbox_language));
    tmnt::localization::SetActiveLanguage(language.key);
    tmnt::PublishTitleMusicLevel(defaults.user_data_root);
    return defaults;
  }
};

REX_DEFINE_APP(tmnt_turtles_in_time_reshelled, TmntNxApp::Create)
