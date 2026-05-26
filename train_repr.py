import argparse
import os
import random

import h5py
import numpy as np
import torch
from tqdm import trange

import algo.utils as utils
from algo.representation.tc_elbo import TCELBO


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_save_dir(args):
    """
    If --save_dir is provided, use it directly.
    Otherwise create a DVDF-style default path.
    """
    if args.save_dir is not None:
        return args.save_dir

    return os.path.join(
        "logs",
        "Repr",
        args.env,
        args.srctype,
        str(args.seed),
        "models",
    )


def main():
    parser = argparse.ArgumentParser()

    # Dataset / logging
    parser.add_argument("--src_path", type=str, default=None)
    parser.add_argument("--env", type=str, default="ant-kinematic")
    parser.add_argument("--srctype", type=str, default="medium-replay")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=None)

    # Buffer / trajectory sampling
    parser.add_argument("--max_size", type=int, default=int(1e6))
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=200)

    # Representation model
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--hidden_sizes", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--ensemble_size", type=int, default=3)

    # Optimization
    parser.add_argument("--repr_steps", type=int, default=100000)
    parser.add_argument("--encoder_lr", type=float, default=3e-4)
    parser.add_argument("--decoder_lr", type=float, default=3e-4)
    parser.add_argument("--beta_kl", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)

    # Logging / checkpoint
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=500)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = make_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    
    args.src_path = f"./dataset/source/{args.env}-{args.srctype}.hdf5"

    print("---------------------------------------")
    print("Training temporally-consistent representation")
    print(f"Env: {args.env}")
    print(f"Source type: {args.srctype}")
    print(f"Source path: {args.src_path}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
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
    # 2. Build TCELBO algorithm
    # ------------------------------------------------------------------
    config = vars(args).copy()
    config["state_dim"] = state_dim
    config["action_dim"] = action_dim

    policy = TCELBO(config, device)

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    for t in trange(int(args.repr_steps)):
        info = policy.train(
            replay_buffer=src_buffer,
            batch_size=args.batch_size,
            max_len=args.max_len,
            writer=None,
        )

        step = t + 1

        if step % args.eval_freq == 0:
            print(
                f"[step {step}] "
                f"enc_loss={info['encoder_loss'].item():.6f}, "
                f"enc_recon={info['encoder_recon_loss'].item():.6f}, "
                f"dec_loss={info['decoder_loss'].item():.6f}, "
                f"kl={info['kl_loss'].item():.6f}"
            )

            # z collapse check
            with torch.no_grad():
                eval_batch = src_buffer.sample_trajectories(
                    batch_size=min(64, args.batch_size * 4),
                    max_len=args.max_len,
                )
                z = policy.infer_z(eval_batch)

                print(
                    f"[z stats] "
                    f"z_abs={z.abs().mean().item():.6f}, "
                    f"z_std={z.std(dim=0).mean().item():.6f}, "
                    f"z_min={z.min().item():.6f}, "
                    f"z_max={z.max().item():.6f}"
                )

        if step % args.save_freq == 0:
            model_path = os.path.join(save_dir, "model")
            policy.save(model_path)
            print(
                f"Saved checkpoint to:\n"
                f"  {model_path}_encoder\n"
                f"  {model_path}_decoder"
            )

    # ------------------------------------------------------------------
    # 4. Save final model
    # ------------------------------------------------------------------
    final_path = os.path.join(save_dir, "model_final")
    policy.save(final_path)

    print("---------------------------------------")
    print("Finished representation training.")
    print(f"Saved final model to:")
    print(f"  {final_path}_encoder")
    print(f"  {final_path}_decoder")
    print("---------------------------------------")


if __name__ == "__main__":
    main()