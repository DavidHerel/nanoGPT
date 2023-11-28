# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

year = 'merged_'
#mil_words = '100mil'
mil_words = '1bil'
out_dir = year+mil_words+'-finetune-2022-'
dataset = '../1_bil_words/'

wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name=year+mil_words+'-gpt2-124M-scratch-'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 3
block_size = 256
gradient_accumulation_steps = 5 * 8

init_from = 'gpt2-xl' # this is the largest GPT-2 model

# # this makes total number of tokens be 300B
# max_iters = 1000
# lr_decay_iters = 1000
#
# # eval stuff
# eval_interval = 1000
# eval_iters = 200
# log_interval = 10

learning_rate = 3e-5
decay_lr = False

# weight decay
weight_decay = 1e-1
