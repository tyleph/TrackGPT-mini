# forecast.py

import os
import numpy as np
import torch
import pickle
from model import GPT, GPTConfig
from contextlib import nullcontext

def test_single_track(model, meta, track_idx, input_length=48, forecast_length=12):
    """Test forecasting on a single validation track"""
    
    # Load validation data
    val_tokens = np.memmap('data/ais/val.bin', dtype=np.uint16, mode='r')
    val_offsets = np.load('data/ais/val_track_offsets.npy')
    
    if track_idx >= len(val_offsets):
        raise ValueError(f"Track index {track_idx} exceeds available tracks ({len(val_offsets)})")
    
    starting, end = val_offsets[track_idx]
    full_track = val_tokens[starting:end]
    
    if len(full_track) < input_length + forecast_length:
        raise ValueError(f"Track too short: {len(full_track)} < {input_length + forecast_length}")
    
    # Split into input and ground truth
    input_tokens = full_track[:input_length]
    ground_truth = full_track[input_length:input_length + forecast_length]
    
    # Generate forecast
    device = next(model.parameters()).device
    x = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        # Generate with temperature=0.92 as in paper
        forecast = model.generate(x, forecast_length, temperature=0.92, top_k=5)
        forecast_tokens = forecast[0, input_length:].cpu().numpy()
    
    return {
        'input_tokens': input_tokens,
        'forecast_tokens': forecast_tokens,
        'ground_truth': ground_truth,
        'track_length': len(full_track)
    }

def decode_tokens_to_geohashes(tokens, fixed_prefix):
    """Decode tokens back to geohashes"""
    base32_chars = '0123456789bcdefghjkmnpqrstuvwxyz'
    int_to_char_map = {i: char for i, char in enumerate(base32_chars)}
    
    def decode_token(token):
        val1 = (token >> 11) & 0b11111
        val2 = (token >> 6) & 0b11111
        val3 = (token >> 1) & 0b11111
        try:
            return fixed_prefix + int_to_char_map[val1] + int_to_char_map[val2] + int_to_char_map[val3]
        except KeyError:
            return fixed_prefix + "???"
    
    return [decode_token(t) for t in tokens]

def get_trajectories(track_idx, input_length, forecast_length):
    """
    Loads the model, runs a forecast for a given track, and returns the geohashes.
    This function is designed to be called from another script or notebook.
    """
    # Load model and metadata
    ckpt_path = 'out-trackgpt/ckpt.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    with open('data/ais/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    # Initialize model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    
    # Test forecast
    result = test_single_track(model, meta, track_idx, input_length, forecast_length)
    
    # Decode and return
    fixed_prefix = meta['fixed_prefix']
    input_geohashes = decode_tokens_to_geohashes(result['input_tokens'], fixed_prefix)
    forecast_geohashes = decode_tokens_to_geohashes(result['forecast_tokens'], fixed_prefix)
    truth_geohashes = decode_tokens_to_geohashes(result['ground_truth'], fixed_prefix)
    
    return input_geohashes, forecast_geohashes, truth_geohashes

# Main execution block remains for standalone script usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python forecast.py <track_index> [input_length] [forecast_length]")
        sys.exit(1)
    
    track_idx = int(sys.argv[1])
    input_length = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    forecast_length = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    # Get trajectories using the new function
    input_g, forecast_g, truth_g = get_trajectories(track_idx, input_length, forecast_length)
    
    print(f"Testing track {track_idx} (input_length={input_length}, forecast_length={forecast_length})")
    print(f"Input geohashes ({len(input_g)}):")
    print(' -> '.join(input_g))
    print(f"\nForecast ({len(forecast_g)}):")
    print(' -> '.join(forecast_g))
    print(f"\nGround truth ({len(truth_g)}):")
    print(' -> '.join(truth_g))
