# Modified action type


## Seed run

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/onx1o14x)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/68nsxz82)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name small_seedpool

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool
```