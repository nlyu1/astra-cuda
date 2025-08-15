# Modified action type

1. In normal game, we observed collapse where explained variance went to 1 while policy loss lagged. 
2. Trying separate critic 

## Seed run

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/nn6509bw)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/10j9lv08)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name small_seedpool --learning_rate 1e-4 > small_seedpool.txt

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool --learning_rate 1e-4 > normal_seedpool.txt
```

## Bootstrap runs 

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/0fpmbq8z)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/pu4qv506)

```
CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 1 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1  --num_iterations 40000000 --exp_name small0 --checkpoint_name small_seedpool_4000

CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 0 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1  --num_iterations 40000000 --exp_name normal0 --checkpoint_name normal_seedpool_4000
```

## Second bootstrap run

Add larger gae_lambda

**Results:**
1. [Small1 game (offline run synced)](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/pg0xwhpg) - Small game with gae_lambda=0.7
2. [Second run (offline run synced)](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/x627sroj) - Normal game continuation

**Local logs location:** `python/checkpoints/logs/wandb/offline-run-20250811_004211-pg0xwhpg/` and `python/checkpoints/logs/wandb/offline-run-20250811_004244-x627sroj/`

```

export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --psettlement_coef 0.2 --gae_lambda 0.7 --proles_coef 0.2  --num_iterations 40000000 --exp_name small1 --checkpoint_name small0_318000

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --psettlement_coef 0.2 --gae_lambda 0.7 --proles_coef 0.2  --num_iterations 40000000 --exp_name normal0 --checkpoint_name normal0_57000
```

## Third bootstrap run 

1. [Small run](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/wllbx15p)
2. [Normal run](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/e7477vc0)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --psettlement_coef 0.2 --gae_lambda 0.9 --proles_coef 0.2 --self_play_prob 0.5 --num_iterations 40000000 --exp_name small2 --checkpoint_name small1_153000

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --psettlement_coef 0.2 --gae_lambda 0.9 --proles_coef 0.2 --self_play_prob 0.5 --num_iterations 40000000 --exp_name normal1 --checkpoint_name normal0_135000
```

Observing steadily increasing spread. 

# Fourth bootstrap run 

Lower entropy coefficient and higher `gae_lambda`

1. [Small run](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/fg8xn2g8)
2. [Normal run](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/p9jgqyz5)
```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.005 --psettlement_coef 0.2 --gae_lambda 0.95 --proles_coef 0.2 --self_play_prob 0.5 --num_iterations 40000000 --exp_name small3 --checkpoint_name small2_180000

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.005 --psettlement_coef 0.2 --gae_lambda 0.95 --proles_coef 0.2 --self_play_prob 0.5 --num_iterations 40000000 --exp_name normal2 --checkpoint_name normal1_156000
```

## How to sync future offline runs

To sync future offline wandb runs to cloud:

1. Activate the astra environment: `conda activate astra`
2. Navigate to python directory: `cd python/`
3. Sync offline runs: `wandb sync checkpoints/logs/wandb/offline-run-YYYYMMDD_HHMMSS-RUNID/`

Local wandb logs are stored in: `python/checkpoints/logs/wandb/`


# Thompson sampling + exploiter

Main takeaways for seed training:

1. Don't add private info modeling. Else training will be unstable. 
2. Don't add self-play. 
3. Use small gae_lambda. 
4. Use lr warmup. All ingredients above are important. 

Initialize with a pool. Changed effective memory size to 1000 to reduce overly sharp sampling behavior. 

```
CUDA_VISIBLE_DEVICES=1 python main_process.py --game_setting 1 --entropy_coef 0.003 --gae_lambda 0.95 --psettlement_coef 0.2 --proles_coef 0.2 --game_setting 1 --num_iterations 4000000000000 --self_play_prob 0.5 --iterations_per_heavy_logging 1500 --benchmark_checkpoint_name poolrun_selfplayonly/small3_129000 --exp_name smallthompson_again --checkpoint_name poolrun_selfplayonly/small3_126000
```

## Thompson-sampling main only

[Run](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/cqlui7xa?nw=nwuserlyuxingjian)
```
CUDA_VISIBLE_DEVICES=0 python main_process.py --game_setting 0 --entropy_coef 0.003 --psettlement_coef 0.2 --gae_lambda 0.95 --proles_coef 0.2 --self_play_prob 0.5 --iterations_per_heavy_logging 1500 --num_iterations 40000000 --exp_name normal3 --checkpoint_name poolrun_selfplayonly/normal2_204000 --benchmark_checkpoint_name poolrun_selfplayonly/normal2_207000
```

## Exploiter process for normal3

Run this in a separate terminal to continuously train exploiters against the latest main checkpoint:

```
CUDA_VISIBLE_DEVICES=1 python exploiter_process.py checkpoints/normal3
```