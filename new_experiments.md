# Modified action type

1. In normal game, we observed collapse where explained variance went to 1 while policy loss lagged. 
2. Trying separate critic 

## Seed run

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/nn6509bw)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/10j9lv08)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name small_seedpool > small_seedpool.txt

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01 --num_iterations 4005 --iterations_per_checkpoint 4000 --iterations_per_pool_update 4000 --exp_name normal_seedpool > normal_seedpool.txt
```

## Bootstrap runs 

1. [Small game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/spe18c10)
2. [Normal game](https://wandb.ai/lyuxingjian-na/HighLowTrading/runs/w3mf6qro)

```
CUDA_VISIBLE_DEVICES=1 python vtrace.py --game_setting 1 --entropy_coef 0.01 --psettlement_coef 0.1 --proles_coef 0.1  --num_iterations 40000000 --exp_name small0 --checkpoint_name small_seedpool_4000 > small.txt

CUDA_VISIBLE_DEVICES=0 python vtrace.py --game_setting 0 --entropy_coef 0.01  --psettlement_coef 0.1 --proles_coef 0.1  --num_iterations 40000000 --exp_name normal0 --checkpoint_name normal_seedpool_4000 > normal.txt
```