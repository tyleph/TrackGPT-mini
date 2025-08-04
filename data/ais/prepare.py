import os
import pickle
import numpy as np
import pandas as pd
import pygeohash as pgh

# --- Constants and Configuration ---
FIXED_PREFIX = 'u'
GEOHASH_BASE32_MAP = {char: i for i, char in enumerate("0123456789bcdefghjkmnpqrstuvwxyz")}
INPUT_CSV_PATH = 'data/ais/ais_preprocessed.csv'
OUTPUT_DIR = os.path.dirname(__file__)

def encode_geohash_to_uint16(gh: str) -> int:
    """
    Encodes a geohash string into a 16-bit integer token using the TrackGPT method.
    """
    if not gh.startswith(FIXED_PREFIX):
        raise ValueError(f"Geohash '{gh}' does not start with the fixed prefix '{FIXED_PREFIX}'")

    suffix = gh[len(FIXED_PREFIX):]
    if len(suffix) < 4:
        suffix = suffix.ljust(4, '0')

    three_chars = suffix[:3]
    four_chars = suffix[:4]

    parent_box = pgh.get_bounding_box(FIXED_PREFIX + three_chars)
    child_box = pgh.get_bounding_box(FIXED_PREFIX + four_chars)
    
    parent_center_lon = (parent_box.max_lon + parent_box.min_lon) / 2
    child_center_lon = (child_box.max_lon + child_box.min_lon) / 2
    east_west_flag = 1 if child_center_lon >= parent_center_lon else 0

    try:
        val1 = GEOHASH_BASE32_MAP[three_chars[0]]
        val2 = GEOHASH_BASE32_MAP[three_chars[1]]
        val3 = GEOHASH_BASE32_MAP[three_chars[2]]
    except KeyError as e:
        raise KeyError(f"Invalid character in geohash suffix '{three_chars}'. Full geohash: '{gh}'.") from e
        
    token = (val1 << 11) | (val2 << 6) | (val3 << 1) | east_west_flag
    
    return token

# --- Main Execution ---

print(f"Reading data from {INPUT_CSV_PATH}...")
df = pd.read_csv(INPUT_CSV_PATH)

print("Encoding geohashes into tokens...")
all_tracks = []
for _, group in df.groupby('sub_track', sort=False):
    geohashes = group['geohash'].tolist()
    tokens = [encode_geohash_to_uint16(gh) for gh in geohashes]
    all_tracks.append(np.array(tokens, dtype=np.uint16))

print(f"Processed {len(all_tracks)} tracks.")

split_idx = int(0.9 * len(all_tracks))
train_tracks = all_tracks[:split_idx]
val_tracks = all_tracks[split_idx:]

# --- NEW: Function to process tracks and create offset files ---
def process_and_save(tracks, split_name):
    """
    Calculates track offsets, concatenates tokens, and saves all necessary files.
    """
    track_offsets = []
    current_offset = 0
    # Calculate the start and end index for each track
    for track in tracks:
        start = current_offset
        end = current_offset + len(track)
        track_offsets.append((start, end))
        current_offset = end
    
    # Save the offsets file
    offsets_array = np.array(track_offsets, dtype=np.int64)
    offsets_filename = os.path.join(OUTPUT_DIR, f'{split_name}_track_offsets.npy')
    np.save(offsets_filename, offsets_array)
    print(f"Saved track offsets to {offsets_filename}")

    # Concatenate all tokens into one large array
    all_tokens = np.concatenate(tracks)
    
    # Save the binary token file
    tokens_filename = os.path.join(OUTPUT_DIR, f'{split_name}.bin')
    all_tokens.tofile(tokens_filename)
    print(f"{split_name.capitalize()} set: {len(all_tokens):,} tokens")

# Process both train and validation splits
process_and_save(train_tracks, 'train')
process_and_save(val_tracks, 'val')

# Save metadata
meta = {
    'vocab_size': 65536,
    'tokenizer': 'custom_geohash16',
    'fixed_prefix': FIXED_PREFIX,
}

with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Metadata saved to meta.pkl with fixed_prefix: '{FIXED_PREFIX}'")