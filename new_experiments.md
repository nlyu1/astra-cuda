# Modified action type


## Seed run

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/651t02yx)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/ksfzenbd)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.03 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name small_seedpool > small_seedpool.txt

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.02 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool > normal_seedpool.txt
```

## Bootstrap runs 

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/kc4tdjwr)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/i8qju8rm)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1  --num_iterations 40000000 --exp_name small0 --checkpoint_name small_seedpool_4000 --learning_rate 3e-4 > small.txt

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1  --num_iterations 40000000 --exp_name normal0 --checkpoint_name normal_seedpool_4000 --learning_rate 3e-4 > normal.txt
```

CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 40000000 --exp_name small0 --checkpoint_name small_seedpool_4000 --num_warmup_steps 0 