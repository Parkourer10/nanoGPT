

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

n_layer = 6
n_head = 6
n_embd = 256
dropout = 0.2


batch_size = 64
block_size = 512
gradient_accumulation_steps = 4

max_iters = 30000
lr_decay_iters = 30000

eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
