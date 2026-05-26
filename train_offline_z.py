import argparse
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm, trange

import algo.utils as utils
from algo.offline.sql_z import SQL_Z


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load_hdf5_dataset(path):
    data_dict = {}

    with h5py.File(path, "r") as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:
                data_dict[k] = dataset_file[k][:]
            except ValueError:
                data_dict[k] = dataset_file[k][()]

    return data_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(env_name):
    """
    Load SQL_Z config in the same style as train_offline.py.

    For env = ant-kinematic:
        src_env_name = ant

    Prefer:
        config/mujoco/sql_z/ant.yaml

    Fallback:
        config/mujoco/sql/ant.yaml
    """
    repo_root = str(Path(__file__).parent.absolute())
    src_env_name = env_name.split("-")[0]

    sql_z_config_path = os.path.join(
        repo_root,
        "config",
        "mujoco",
        "sql_z",
        f"{src_env_name}.yaml",
    )

    sql_config_path = os.path.join(
        repo_root,
        "config",
        "mujoco",
        "iql",
        f"{src_env_name}.yaml",
    )

    if os.path.exists(sql_z_config_path):
        config_path = sql_z_config_path
    else:
        config_path = sql_config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")
    return config


def override_config(config, args):
    """
    Explicit command-line overrides.

    Rule:
        If an argument is not None, override yaml config.
        If it is None, keep yaml value.

    This avoids JSON-style --params and keeps all tunable options visible.
    """
    override_keys = [
        "gamma",
        "tau",
        "update_interval",
        "alpha",
        "critic_lr",
        "hidden_sizes",
        "batch_size",
        "max_step",
        "eval_freq",
    ]

    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    return config


def main():
    parser = argparse.ArgumentParser()

    # Basic setting, same style as train_offline.py
    parser.add_argument("--dir", default="./logs/OfflineZ")
    parser.add_argument("--env", default="ant-kinematic")
    parser.add_argument("--srctype", default="medium-replay")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    # Optional explicit path override.
    # If not provided, use fixed DVDF-style paths below.
    parser.add_argument("--src_path", default=None, type=str)
    parser.add_argument("--z_path", default=None, type=str)

    # Buffer setting
    parser.add_argument("--max_size", default=int(1e6), type=int)
    parser.add_argument("--max_episode_steps", default=1000, type=int)

    # Explicit config overrides.
    # Default None means: use yaml config.
    parser.add_argument("--gamma", default=None, type=float)
    parser.add_argument("--tau", default=None, type=float)
    parser.add_argument("--update_interval", default=None, type=int)
    parser.add_argument("--alpha", default=None, type=float)

    parser.add_argument("--critic_lr", default=None, type=float)
    parser.add_argument("--hidden_sizes", default=None, type=int)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_step", default=5000, type=int)
    parser.add_argument("--eval_freq", default=500, type=int)

    # Save
    parser.add_argument("--save_model", default=True, type=str2bool)

    args = parser.parse_args()

    if "_" in args.env:
        args.env = args.env.replace("_", "-")

    # ------------------------------------------------------------
    # Fixed DVDF-style paths
    # ------------------------------------------------------------
    if args.src_path is None:
        args.src_path = f"./dataset/source/{args.env}-{args.srctype}.hdf5"

    if args.z_path is None:
        args.z_path = f"./logs/Repr/{args.env}/{args.srctype}/{args.seed}/models/source_z.npz"

    args.outdir = f"{args.dir}/{args.env}/{args.srctype}/{args.seed}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    if args.save_model:
        os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)

    print("------------------------------------------------------------")
    print("Training modality-aware SQL_Z")
    print(f"Env: {args.env}")
    print(f"Source type: {args.srctype}")
    print(f"Seed: {args.seed}")
    print(f"Source path: {args.src_path}")
    print(f"Z path: {args.z_path}")
    print(f"Output dir: {args.outdir}")
    print(f"Device: {device}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # 1. Load source dataset and z labels
    # ------------------------------------------------------------
    src_dataset = load_hdf5_dataset(args.src_path)

    z_data = np.load(args.z_path)
    zs = z_data["zs"].astype(np.float32)

    state_dim = src_dataset["observations"].shape[1]
    action_dim = src_dataset["actions"].shape[1]
    z_dim = zs.shape[1]

    assert zs.shape[0] == src_dataset["observations"].shape[0], (
        f"z size mismatch: zs has {zs.shape[0]} transitions, "
        f"dataset has {src_dataset['observations'].shape[0]} transitions."
    )

    # ------------------------------------------------------------
    # 2. Load yaml config and override explicitly
    # ------------------------------------------------------------
    config = load_config(args.env)
    config = override_config(config, args)

    config.update({
        "env_name": args.env,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "z_dim": z_dim,
    })

    # Safe defaults if old sql yaml does not contain these.
    if "update_interval" not in config:
        config["update_interval"] = 1
    if "eval_freq" not in config:
        config["eval_freq"] = 5000

    print("------------------------------------------------------------")
    print("Config:")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("------------------------------------------------------------")

    with open(os.path.join(args.outdir, "log.txt"), "w") as f:
        f.write(f"Policy: SQL_Z\n")
        f.write(f"Dataset: {args.env}-{args.srctype}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"src_path: {args.src_path}\n")
        f.write(f"z_path: {args.z_path}\n")
        f.write("\nConfig:\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    # ------------------------------------------------------------
    # 3. Build source buffer with z
    # ------------------------------------------------------------
    src_replay_buffer = utils.SourceTrajectoryBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        max_size=args.max_size,
        max_episode_steps=args.max_episode_steps,
    )

    src_replay_buffer.convert_D4RL(src_dataset)
    src_replay_buffer.set_zs(zs)

    print("Source trajectory buffer summary:")
    print(src_replay_buffer.summary())
    print(f"Loaded z shape: {zs.shape}")
    print(f"z_abs_mean: {np.mean(np.abs(zs)):.6f}")
    print(f"z_std_mean: {np.std(zs, axis=0).mean():.6f}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # 4. Build SQL_Z and train
    # ------------------------------------------------------------
    policy = SQL_Z(config, device)

    for t in trange(int(config["max_step"])):
        info = policy.train(
            src_replay_buffer,
            config["batch_size"],
            writer=None,
        )

        step = t + 1

        if step % int(config["eval_freq"]) == 0:
            print(
                f"Step: {step} "
                f"v_loss: {info['v_loss'].item():.6f} "
                f"q_loss: {info['q_loss'].item():.6f} "
                f"adv_mean: {info['adv_mean'].item():.6f} "
                f"adv_std: {info['adv_std'].item():.6f}"
            )

            with open(os.path.join(args.outdir, "loss.txt"), "a") as f:
                f.write(
                    f"{step} "
                    f"{info['v_loss'].item()} "
                    f"{info['q_loss'].item()} "
                    f"{info['adv_mean'].item()} "
                    f"{info['adv_std'].item()}\n"
                )

            if args.save_model:
                policy.save(os.path.join(args.outdir, "models", "model"))

    if args.save_model:
        policy.save(os.path.join(args.outdir, "models", "model_final"))

    print("------------------------------------------------------------")
    print("Finished SQL_Z training.")
    print("Saved models:")
    print(os.path.join(args.outdir, "models", "model_final_critic"))
    print(os.path.join(args.outdir, "models", "model_final_value"))
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()