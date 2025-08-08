# Modified action type

Switching to kuma distribution. 

## Seed run

Swept over learning-rate (1e-4, 3e-4, 5e-4) and entropy coefficient (0.01, 0.05, 0.1)

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/29ohrrg8)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/pvnakp1i)

```
CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --num_iterations 5010 --iterations_per_checkpoint 5000 --iterations_per_pool_update 5000 --exp_name normal_seedpool --learning_rate 3e-4
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 5010 --iterations_per_checkpoint 5000 --iterations_per_pool_update 5000 --exp_name small_seedpool --learning_rate 3e-4
```

## Bootstrap run

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/jy2rt05u)
1. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/uxlmmvjc)

```
CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1 --num_iterations 300000000 --exp_name small_pool0 --checkpoint_name small_seedpool7_5000

CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1 --num_iterations 300000000 --exp_name small_pool0 --checkpoint_name small_seedpool7_5000
```

```
CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1 --num_iterations 300000000 --exp_name debug_nans --checkpoint_name small_pool0_93000
```