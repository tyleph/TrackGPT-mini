"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import re
import numpy as np

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 24 # number of tokens generated in each sample
temperature = 0.92 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 5 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    if meta['tokenizer'] == 'custom_geohash16':
        vocab_size = meta['vocab_size']
        
        # In TrackGPT mode, encoding is just int -> int
        encode = lambda s: np.array(s, dtype=np.uint16)
        decode = lambda l: np.array(l, dtype=np.uint16)
    else:
        enc = tiktoken.get_encoding(meta['tokenizer'])
        # enc = meta['tokenizer']
        stoi = { enc.decode_single_token_bytes(i): i for i in range(enc.n_vocab) }
        encode = lambda s: enc.encode_ordinary(s)
        decode = lambda l: enc.decode(l)
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)




# Add these parameters after your existing ones:
start_track_idx = None  # Track index to use for testing
input_length = 20      # How many tokens to use as input
# Add to configurator overrides

# Replace your hard-coded track loading with:
if start_track_idx is not None:
    # Load validation data (not training!)
    val_tokens = np.memmap('data/ais/val.bin', dtype=np.uint16, mode='r')
    val_offsets = np.load('data/ais/val_track_offsets.npy')
    
    if start_track_idx >= len(val_offsets):
        print(f"Error: Track index {start_track_idx} exceeds available tracks ({len(val_offsets)})")
        exit(1)
    
    starting, end = val_offsets[start_track_idx]
    example_tokens = val_tokens[starting:end]
    
    if len(example_tokens) < input_length + 10:  # Need some buffer for target
        print(f"Warning: Track {start_track_idx} too short ({len(example_tokens)} tokens)")
    
    # Use only first part as input for forecasting
    start_tokens = example_tokens[:input_length]
    target_tokens = example_tokens[input_length:]  # For comparison
    
    print(f"Using validation track {start_track_idx}")
    print(f"Input length: {len(start_tokens)}, Target length: {len(target_tokens)}")
    
else:
    # Default behavior or error
    print("Please specify --start_track_idx=<number> to test a specific validation track")
    exit(1)







# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
# start_ids = encode(start)
start_ids = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
if isinstance(start_ids, torch.Tensor):
    x = start_ids.clone().detach().to(dtype=torch.long, device=device)  # if already a tensor
else:
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # if still numpy



# --- Decoder Setup ---

# Create the inverse mapping from integer to character for decoding.
# This is the reverse of the map used in prepare.py.
base32_chars = '0123456789bcdefghjkmnpqrstuvwxyz'
int_to_char_map = {i: char for i, char in enumerate(base32_chars)}

# Load the single fixed prefix from the metadata file.
# This is now the correct way to get the prefix.
fixed_prefix = meta['fixed_prefix']

def decode_token_to_suffix(token: int) -> str:
    """
    Decodes a 16-bit integer token back into its 3-character geohash suffix.
    
    This function correctly reverses the bitwise encoding:
    token = (val1 << 11) | (val2 << 6) | (val3 << 1) | east_west_flag
    """
    # Use bitwise operations to extract the original 5-bit integer values.
    # 0b11111 (or 31) is a mask to select the 5 relevant bits.
    val1 = (token >> 11) & 0b11111
    val2 = (token >> 6)  & 0b11111
    val3 = (token >> 1)  & 0b11111
    
    # The last bit is the east/west flag, which we can extract but will ignore
    # for simple geohash string reconstruction, just as the paper implies by
    # decoding to geohash tracks.
    # ew_flag = token & 0b1

    # Convert the integer values back to characters using the inverse map.
    try:
        char1 = int_to_char_map[val1]
        char2 = int_to_char_map[val2]
        char3 = int_to_char_map[val3]
        suffix = char1 + char2 + char3
        return suffix
    except KeyError as e:
        # Handle the case where a token might be out of the expected range.
        print(f"Warning: Decoding error for token {token}, value {e} out of map range.")
        return "<?>"


# --- Generation Loop ---
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # Generate a sequence of new tokens from the model.
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            # Get the generated tokens as a numpy array.
            sampled_tokens = y[0].cpu().numpy()

            print("--- GENERATED SEQUENCE ---")
            
            # Decode all generated tokens into geohash strings.
            # This combines the fixed prefix with the decoded 3-character suffix.
            decoded_geohashes = [fixed_prefix + decode_token_to_suffix(t) for t in sampled_tokens]

            # Print the full, human-readable geohash sequence.
            for gh in decoded_geohashes:
                print(gh)
            print('--------------------------')