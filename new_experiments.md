# Modified action type


## Seed run

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/651t02yx)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/ksfzenbd)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name small_seedpool_ > small_seedpool.txt

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool > normal_seedpool.txt
```

##

CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool_dev --checkpoint_name small_seedpool_4000 --warmup_steps 0 > logs.txt