# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
year = '2019-08'
mil_words = '3bil'
out_dir = mil_words+'-scratch-2019-'+year
dataset = '../3_bil_words/'+year

wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name=mil_words+'-nanogpt-3bil-scratch-'+year

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 200000
lr_decay_iters = 200000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1