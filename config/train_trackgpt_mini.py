# adapted the character-level shakespeare model for training trackgpt on a small dataset
# good for debugging and playing on macbooks and such

out_dir = 'out-trackgpt'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'trackgpt'
wandb_run_name = 'mini-gpt'

dataset = 'ais'
gradient_accumulation_steps = 4
batch_size = 64
block_size = 48 # context of up to __ previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.07

learning_rate = 2e-4 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 200 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
