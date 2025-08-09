# Modified action type


## Seed run

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/4p2d5ioe)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/r4am7n93)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name small_seedpool_ > small_seedpool.txt

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool_ > normal_seedpool.txt
```

##

CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool_dev --checkpoint_name small_seedpool_4000 > logs.txt