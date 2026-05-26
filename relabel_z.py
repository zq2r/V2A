import argparse
import os
import random

import h5py
import numpy as np
import torch
from tqdm import tqdm

import algo.utils as utils
from algo.representation.tc_elbo import TCELBO


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_single_trajectory_batch(traj, device):
    """
    Convert one trajectory from SourceTrajectoryBuffer.iter_trajectories()
    to the batch format required by TCELBO.infer_z().

    Input traj fields are numpy arrays:
        observations:      [T, state_dim]
        actions:           [T, action_dim]
        next_observations: [T, state_dim]

    Output batch fields are torch tensors:
        observations:      [1, T, state_dim]
        actions:           [1, T, action_dim]
        next_observations: [1, T, state_dim]
        mask:              [1, T, 1]
    """
    observations = traj["observations"]
    actions = traj["actions"]
    next_observations = traj["next_observations"]

    traj_len = observations.shape[0]

    batch = {
        "observations": torch.FloatTensor(observations[None]).to(device),
        "actions": torch.FloatTensor(actions[None]).to(device),
        "next_observations": torch.FloatTensor(next_observations[None]).to(device),
        "mask": torch.ones((1, traj_len, 1), dtype=torch.float32).to(device),
    }

    return batch


def check_z_consistency(src_buffer, zs, num_check=20):
    """
    Sanity check:
        all transitions in the same trajectory should share exactly the same z.
    """
    max_diff = 0.0

    for i, traj in enumerate(src_buffer.iter_trajectories()):
        if i >= num_check:
            break

        idx = traj["indices"]
        z_traj = zs[idx]

        diff = np.max(np.abs(z_traj - z_traj[0]))
        max_diff = max(max_diff, float(diff))

    print(f"[Sanity Check] max z diff within checked trajectories: {max_diff:.8f}")

    if max_diff > 1e-6:
        raise RuntimeError(
            "Z consistency check failed: transitions in the same trajectory "
            "do not share the same z."
        )


def main():
    parser = argparse.ArgumentParser()

    # Dataset / checkpoint
    parser.add_argument("--src_path", type=str, default=None)
    parser.add_argument("--repr_model_path", type=str, default=None)
    parser.add_argument("--z_save_path", type=str, default=None)

    # Metadata, mostly for logging
    parser.add_argument("--env", type=str, default="ant-kinematic")
    parser.add_argument("--srctype", type=str, default="medium-replay")
    parser.add_argument("--seed", type=int, default=0)

    # Buffer
    parser.add_argument("--max_size", type=int, default=int(1e6))
    parser.add_argument("--max_episode_steps", type=int, default=1000)

    # Must match train_repr.py
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--hidden_sizes", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--encoder_lr", type=float, default=3e-4)
    parser.add_argument("--decoder_lr", type=float, default=3e-4)
    parser.add_argument("--beta_kl", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)

    args = parser.parse_args()
    
    args.src_path= f"./dataset/source/{args.env}-{args.srctype}.hdf5"
    args.repr_model_path = f"./logs/Repr/{args.env}/{args.srctype}/{args.seed}/models/model_final"
    args.z_save_path = f"./logs/Repr/{args.env}/{args.srctype}/{args.seed}/models/source_z.npz"

    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.z_save_path), exist_ok=True)

    print("---------------------------------------")
    print("Relabeling source dataset with trajectory-level z")
    print(f"Env: {args.env}")
    print(f"Source type: {args.srctype}")
    print(f"Source path: {args.src_path}")
    print(f"Representation model path prefix: {args.repr_model_path}")
    print(f"Z save path: {args.z_save_path}")
    print(f"Device: {device}")
    print("---------------------------------------")

    # ------------------------------------------------------------------
    # 1. Load source dataset and build trajectory buffer
    # ------------------------------------------------------------------
    with h5py.File(args.src_path, "r") as dataset:
        state_dim = dataset["observations"].shape[1]
        action_dim = dataset["actions"].shape[1]

        src_buffer = utils.SourceTrajectoryBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            max_size=args.max_size,
            max_episode_steps=args.max_episode_steps,
        )
        src_buffer.convert_D4RL(dataset)

    print("Source trajectory buffer summary:")
    print(src_buffer.summary())

    # ------------------------------------------------------------------
    # 2. Build TCELBO and load trained encoder
    # ------------------------------------------------------------------
    config = vars(args).copy()
    config["state_dim"] = state_dim
    config["action_dim"] = action_dim

    policy = TCELBO(config, device)
    policy.load(args.repr_model_path, load_optimizer=False)
    policy.encoder.eval()

    # ------------------------------------------------------------------
    # 3. Infer one z for each trajectory and assign to all transitions
    # ------------------------------------------------------------------
    zs = np.zeros((src_buffer.size, args.z_dim), dtype=np.float32)
    transition_traj_ids = -np.ones((src_buffer.size,), dtype=np.int64)

    with torch.no_grad():
        for traj in tqdm(
            src_buffer.iter_trajectories(),
            total=src_buffer.num_trajectories,
            desc="Inferring trajectory z",
        ):
            batch = make_single_trajectory_batch(traj, device)

            z = policy.infer_z(batch)
            z = z.squeeze(0).detach().cpu().numpy().astype(np.float32)

            idx = traj["indices"]
            zs[idx] = z
            transition_traj_ids[idx] = traj["traj_id"]

    # ------------------------------------------------------------------
    # 4. Sanity checks
    # ------------------------------------------------------------------
    if np.any(transition_traj_ids < 0):
        num_missing = int(np.sum(transition_traj_ids < 0))
        raise RuntimeError(f"{num_missing} transitions were not assigned trajectory ids.")

    if np.isnan(zs).any():
        raise RuntimeError("NaN detected in inferred zs.")

    check_z_consistency(src_buffer, zs, num_check=20)

    print("---------------------------------------")
    print("Z statistics:")
    print(f"zs.shape: {zs.shape}")
    print(f"z_abs_mean: {np.mean(np.abs(zs)):.6f}")
    print(f"z_std_mean: {np.mean(np.std(zs, axis=0)):.6f}")
    print(f"z_min: {np.min(zs):.6f}")
    print(f"z_max: {np.max(zs):.6f}")
    print("---------------------------------------")

    # ------------------------------------------------------------------
    # 5. Save relabeled z
    # ------------------------------------------------------------------
    np.savez(
        args.z_save_path,
        zs=zs,
        transition_traj_ids=transition_traj_ids,
        traj_lengths=src_buffer.traj_lengths,
        z_dim=np.array([args.z_dim], dtype=np.int64),
        state_dim=np.array([state_dim], dtype=np.int64),
        action_dim=np.array([action_dim], dtype=np.int64),
    )

    print(f"Saved z labels to: {args.z_save_path}")
    print("Done.")


if __name__ == "__main__":
    main()