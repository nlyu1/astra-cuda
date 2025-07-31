# High-Low Trading Game

## Game Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `steps_per_player` | 20 | Number of trading rounds each player gets |
| `max_contracts_per_trade` | 5 | Maximum contracts that can be traded in a single order |
| `customer_max_size` | 5 | Maximum absolute value of target position for customers |
| `max_contract_value` | 30 | Maximum value a contract can have (also used for penalties) |
| `players` | 5 | Number of players in the game (minimum 4) |
| `num_markets` | 32768 | Number of parallel games to run |
| `threads_per_block` | 128 | CUDA threads per block |
| `device_id` | 0 | GPU device to use |

## Game Rules

### Setup Phase
1. **Contract Values**: Two candidate values are randomly drawn from [1, max_contract_value]
2. **Settlement Type**: Either "High" or "Low" is randomly chosen
3. **Final Settlement Value**: 
   - If "High": max(value1, value2)
   - If "Low": min(value1, value2)
4. **Player Roles**: Players are randomly assigned one of three roles:
   - **ValueCheaters** (2 players): Each knows one of the two candidate contract values
   - **HighLowCheater** (1 player): Knows whether settlement will be "High" or "Low"
   - **Customers** (remaining players): Have target positions they must achieve

### Trading Phase
- Players take turns submitting two-sided quotes: (bid_price, bid_size, ask_price, ask_size)
- Orders are matched through a continuous double auction
- Trades execute immediately when orders cross (bid_price >= ask_price)
- Execution price is determined by the resting order (order with lower transaction ID)
- Each player gets `steps_per_player` turns to trade

### Information Structure
Each player observes:
- Their own role and private information
- All players' current positions (contracts held and cash)
- Current market state (best bid/offer and sizes)
- Last traded price
- Historical quotes from all players in current round

## Reward Structure

### Immediate Rewards
Calculated after each action:
```
immediate_reward = cash_diff + customer_progress_reward
```

Where:
- `cash_diff`: Change in cash position from trades
- `customer_progress_reward`: For customers only, `(previous_distance - current_distance) * max_contract_value`
  - `previous_distance`: |previous_position - target_position|
  - `current_distance`: |current_position - target_position|

At terminal state, adds:
- `position * settlement_value`: Value of contracts held at settlement

### Final Returns
Calculated only at game termination:
```
final_return = cash + (contracts * settlement_value) - customer_penalty
```

Where:
- `customer_penalty`: For customers only, `|final_position - target_position| * max_contract_value`

### Key Differences
- **Immediate rewards** track incremental progress and are used for reinforcement learning
- **Final returns** represent the actual game payoff including penalties
- Customers receive positive immediate rewards for moving toward their target but face penalties in final returns for missing it

## Market Mechanics

### Order Matching
- Orders remain in the book until matched or game ends
- Price-time priority: Orders at better prices match first; within same price, earlier orders match first
- Partial fills are supported
- No order cancellation or modification
- Self-trades are allowed

### Portfolio Tracking
Each player maintains:
- Contract position (can be positive or negative)
- Cash position (can be positive or negative)
- Initial portfolio: [0 contracts, 0 cash]

## Observation Tensor

Each player's observation is a float tensor containing:
1. **Role encoding** (3 values): one-hot [is_value_cheater, is_high_low_cheater, is_customer]
2. **Player ID** (2 values): [sin(2π*id/players), cos(2π*id/players)]
3. **Private information** (1 value): contract_value OR high_low_signal OR target_position
4. **All players' quotes and positions** (6 * players values): [bid_px, ask_px, bid_sz, ask_sz, contracts, cash]
5. **Market state** (5 values): [best_bid_px, best_ask_px, best_bid_sz, best_ask_sz, last_trade_px]

Total size: `num_markets × (6 + 6*players + 5)`