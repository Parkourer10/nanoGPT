out_dir = 'out-riyal modal'
eval_interval = 1000
eval_iters = 100
log_interval = 100
always_save_checkpoint = True

# Weights & Biases logging
wandb_log = False
wandb_project = 'gpt-60m'
wandb_run_name = 'gpt-60m'

# Dataset config
dataset = 'tinystories'
gradient_accumulation_steps = 4
batch_size = 64  # Reduce batch size if VRAM is insufficient
block_size = 512

# Model architecture (for ~60M params)
n_layer = 12
n_head = 16
n_embd = 1024  # Hidden size
dropout = 0.1

# Optimization
learning_rate = 3e-4
max_iters = 30000  # Increase for better convergence
lr_decay_iters = 30000
min_lr = 3e-5
beta2 = 0.95
warmup_iters = 1000
device = 'cuda'
compile = True
