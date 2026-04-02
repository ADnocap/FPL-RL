#!/usr/bin/env python3
"""Debug evaluation — trace model decisions to find why it never transfers/uses chips."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sb3_contrib import MaskablePPO
from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.env.fpl_env import FPLEnv, PRESEASON_STEPS
from fpl_rl.env.action_space import (
    ACTION_DIMS, MAX_TRANSFERS_PER_STEP, NUM_TRANSFERS_DIM,
    CHIP_DIM, CHIP_INDEX_MAP, MASK_LENGTH,
)

model = MaskablePPO.load("runs/best_model.zip", device="cpu")
env = FPLEnv(season="2023-24", data_dir=DEFAULT_DATA_DIR)

obs, info = env.reset(seed=42)

# Skip preseason
for i in range(PRESEASON_STEPS):
    masks = env.action_masks()
    action, _ = model.predict(obs, deterministic=True, action_masks=masks)
    obs, reward, term, trunc, info = env.step(action)
    n_transfers = int(action[0])
    print(f"Preseason {i+1}: num_transfers={n_transfers}, FTs remaining={env.state.free_transfers}")

# Run first 5 real GWs with detailed logging
for gw_step in range(5):
    masks = env.action_masks()
    action, _ = model.predict(obs, deterministic=True, action_masks=masks)

    gw = env.state.current_gw
    ft = env.state.free_transfers
    pool_size = len(env.action_encoder._candidate_pool)

    # Parse action
    n_transfers = int(action[0])
    base = 1 + MAX_TRANSFERS_PER_STEP * 2
    captain_idx = int(action[base])
    vice_idx = int(action[base + 1])
    formation_idx = int(action[base + 2])
    chip_idx = int(action[base + 6])

    # Check mask for num_transfers dimension
    transfer_mask = masks[:NUM_TRANSFERS_DIM]
    # Check chip mask
    chip_offset = MASK_LENGTH - CHIP_DIM
    chip_mask = masks[chip_offset:chip_offset + CHIP_DIM]

    # Check transfer_in masks (are pool slots available?)
    # First transfer_in starts at offset 1 + 15 (transfer_out_1) = 16
    first_in_offset = 1 + 15  # after num_transfers + transfer_out_1
    first_in_mask = masks[first_in_offset:first_in_offset + 50]

    print(f"\nGW{gw}: FTs={ft}, pool_size={pool_size}")
    print(f"  Action: num_transfers={n_transfers}, captain={captain_idx}, "
          f"vice={vice_idx}, formation={formation_idx}, chip={chip_idx}")
    print(f"  Transfer dim mask: {transfer_mask} (valid options: {np.where(transfer_mask)[0]})")
    print(f"  Chip mask: {chip_mask} (valid: {[(i, CHIP_INDEX_MAP[i]) for i in np.where(chip_mask)[0]]})")
    print(f"  Pool slots available: {first_in_mask.sum()}/{len(first_in_mask)}")
    print(f"  Bank: {env.state.bank}")

    # Decode to see what engine action looks like
    engine_action = env.action_encoder.decode(action, env.state)
    print(f"  Decoded: transfers_out={engine_action.transfers_out}, "
          f"transfers_in={engine_action.transfers_in}, chip={engine_action.chip}")

    # Check available chips
    chips = env.state.chips
    print(f"  Chips available: WC1={chips.wildcard[0]} WC2={chips.wildcard[1]} "
          f"FH1={chips.free_hit[0]} FH2={chips.free_hit[1]} "
          f"BB1={chips.bench_boost[0]} BB2={chips.bench_boost[1]} "
          f"TC1={chips.triple_captain[0]} TC2={chips.triple_captain[1]}")

    # Step
    obs, reward, term, trunc, info = env.step(action)
    print(f"  Result: gw_pts={info['gw_points']}, net={info['net_points']}, "
          f"total={info['total_points']}")

    if term:
        break

# Also check: what does the action probability distribution look like?
print("\n--- Action distribution analysis ---")
obs_reset, _ = env.reset(seed=42)
# Skip preseason
for _ in range(PRESEASON_STEPS):
    m = env.action_masks()
    a, _ = model.predict(obs_reset, deterministic=True, action_masks=m)
    obs_reset, _, _, _, _ = env.step(a)

# At GW1 (real), get action probabilities
import torch
obs_tensor = torch.tensor(obs_reset, dtype=torch.float32).unsqueeze(0)
masks_tensor = torch.tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0)

with torch.no_grad():
    dist = model.policy.get_distribution(obs_tensor, action_masks=masks_tensor)
    # Get probabilities for num_transfers dimension
    if hasattr(dist, 'distributions'):
        # MultiCategorical
        num_transfer_probs = dist.distributions[0].probs[0].numpy()
        print(f"P(num_transfers) at GW1: {dict(enumerate(num_transfer_probs.round(4)))}")

        chip_probs = dist.distributions[-1].probs[0].numpy()
        print(f"P(chip) at GW1: {dict(enumerate(chip_probs.round(4)))}")
    else:
        print(f"Distribution type: {type(dist)}")
