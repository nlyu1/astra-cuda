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

### Detailed Example: Round 3 Trade

From the playthrough, let's analyze the Round 3 trade where Player 2 quotes 7 @ 21 [2 x 4]:

**Setup:**
- Contract values: 10, 25 (High settlement → final value = 25)
- Player 0 (Customer): Target position = 4
- Player 2 (HighLowCheater): Knows settlement is "High"
- Player 0 had a resting order: 26 @ 27 [3 x 5]

**Trade Execution:**
- Player 2's ask at 21 crosses with Player 0's bid at 26
- Execution price = 26 (the resting order price)
- Fill size = min(3, 4) = 3 contracts

**Immediate Rewards Calculation:**
```python
# Player 0 (Customer):
cash_diff = -3 * 26 = -78
previous_distance = |0 - 4| = 4
current_distance = |3 - 4| = 1
customer_progress = (4 - 1) * 30 = 90
immediate_reward[0] = -78 + 90 = 12

# Player 2 (HighLowCheater):
cash_diff = 3 * 26 = 78
customer_progress = 0 (not a customer)
immediate_reward[2] = 78 + 0 = 78
```

Result: `tensor([12., 0., 78., 0.])`

### Final Returns Example

At game end from the playthrough:
```
Final positions:
Player 0: [5 contracts, -103 cash]
Player 1: [3 contracts, -69 cash]
Player 2: [-5 contracts, 125 cash]
Player 3: [-3 contracts, 47 cash]

Settlement value: 25

Returns calculation:
Player 0: -103 + 5*25 - |5-4|*30 = -103 + 125 - 30 = -8
Player 1: -69 + 3*25 - 0 = -69 + 75 = 6
Player 2: 125 + (-5)*25 - 0 = 125 - 125 = 0
Player 3: 47 + (-3)*25 - 0 = 47 - 75 = -28
```

Note how Player 0 achieved a positive cumulative immediate reward (+12) from getting closer to their target, but still ended with a negative return (-8) due to the penalty for overshooting by 1 contract.

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

### Example from Playthrough
From the playthrough.txt, here's Player 2's observation in Round 3:
```
Player 2 observation tensor (first 6 elements):
  [is_value, is_highlow, is_customer, sin(id), cos(id), private_info]
  tensor([ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.2246e-16, -1.0000e+00,
         1.0000e+00])
```

This decodes as:
- **Role**: HighLowCheater (is_highlow = 1.0)
- **Player ID**: Player 2 encoded as sin(2π*2/4) ≈ 0, cos(2π*2/4) = -1
- **Private info**: 1.0 (knows settlement will be "High")

## Information State String

The information state string provides a human-readable view of what each player knows:

### Private Information Section
Shows player-specific knowledge based on role:
- **ValueCheater**: "Candidate contract value: X"
- **HighLowCheater**: "Settlement will be: High/Low"
- **Customer**: "My target position: X"

### Public Information Section
Includes all publicly observable game state:
- Game configuration parameters
- All players' current positions
- Recent fills with detailed transaction info
- Historical quotes from all players
- Market movement timeline showing BBO evolution
- Current order book state

### Example from Playthrough
From Round 3, Player 2 (HighLowCheater) sees:
```
********** Private Information **********
My role: HighLowCheater
Settlement will be: High
******************************************

********** Player Positions **********
Player 0 position: [0 contracts, 0 cash]
Player 1 position: [0 contracts, 0 cash]
Player 2 position: [0 contracts, 0 cash]
Player 3 position: [0 contracts, 0 cash]
**************************************
```

After their trade executes:
```
Player 2 quotes: 7 @ 21 [2 x 4]
Immediate rewards: tensor([12.,  0., 78.,  0.])
```

This trade crossed with Player 0's resting order at 26, executing at the quote price (26).