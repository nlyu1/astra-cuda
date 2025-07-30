#include "core.h"
#include "high_low_trading/high_low_trading.h"

namespace astra {

void RegisterGames() {
    // Manually register all games here
    REGISTER_ASTRA_GAME(high_low_trading::kGameType, high_low_trading::Factory);
}

}