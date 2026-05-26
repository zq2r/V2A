import argparse
import json
import os
import random
from pathlib import Path

import d4rl
import gym
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

import algo.utils as utils
from algo.call_algo import call_algo


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env
    avg_reward = 0.0

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state = next_state

    avg_reward /= eval_episodes
    print("[{}] Evaluation over {} episodes: {}".format(
        eval_cnt,
        eval_episodes,
        avg_reward,
    ))

    return avg_reward


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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_v2a_config(args, task):
    repo_root = str(Path(__file__).parent.absolute())

    v2a_config_path = f"{repo_root}/config/mujoco/v2a_igdf/{task}.yaml"
    dvdf_config_path = f"{repo_root}/config/mujoco/dv_igdf/{task}.yaml"

    if os.path.exists(v2a_config_path):
        config_path = v2a_config_path
    else:
        config_path = dvdf_config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")

    return config


def override_config(config, args):
    """
    Explicit overrides. None means keeping yaml value.
    """
    override_keys = [
        "gamma",
        "tau",
        "update_interval",
        "lam",
        "temp",
        "actor_lr",
        "critic_lr",
        "hidden_sizes",
        "batch_size",
        "eval_freq",
        "info_update_step",
        "repr_dim",
        "ensemble_size",
    ]

    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default="./logs/V2A")
    parser.add_argument("--algo", default="V2A_IGDF")
    parser.add_argument("--env", default="ant-kinematic")
    parser.add_argument("--srctype", default="medium-replay")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--save-model", default=True, type=str2bool)
    parser.add_argument("--limited_size", default=False, type=str2bool)
    parser.add_argument("--tar_env_interact_interval", default=10, type=int)

    parser.add_argument("--max_step", default=10000, type=int)
    parser.add_argument("--target_ratio", default=0.1, type=float)

    # V2A filtering hyperparameters
    parser.add_argument("--tradeoff", default=0.6, type=float)
    parser.add_argument("--xi", default=0.5, type=float)

    # Optional explicit config overrides
    parser.add_argument("--gamma", default=None, type=float)
    parser.add_argument("--tau", default=None, type=float)
    parser.add_argument("--update_interval", default=None, type=int)
    parser.add_argument("--lam", default=None, type=float)
    parser.add_argument("--temp", default=None, type=float)
    parser.add_argument("--actor_lr", default=None, type=float)
    parser.add_argument("--critic_lr", default=None, type=float)
    parser.add_argument("--hidden_sizes", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--info_update_step", default=None, type=int)
    parser.add_argument("--repr_dim", default=None, type=int)
    parser.add_argument("--ensemble_size", default=None, type=int)

    args = parser.parse_args()

    if "_" in args.env:
        args.env = args.env.replace("_", "-")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo_root = str(Path(__file__).parent.absolute())

    task = args.env.split("-")[0]

    set_seed(args.seed)

    # ------------------------------------------------------------
    # Fixed DVDF-style paths
    # ------------------------------------------------------------
    src_dataset_path = f"{repo_root}/dataset/source/{args.env}-{args.srctype}.hdf5"
    z_path = f"{repo_root}/logs/Repr/{args.env}/{args.srctype}/{args.seed}/models/source_z.npz"

    src_Q_z_path = f"{repo_root}/logs/OfflineZ/{args.env}/{args.srctype}/{args.seed}/models/model_final_critic"
    src_V_z_path = f"{repo_root}/logs/OfflineZ/{args.env}/{args.srctype}/{args.seed}/models/model_final_value"

    outdir = f"{args.dir}/{args.env}/{args.srctype}/{args.seed}"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.save_model and not os.path.exists(f"{outdir}/models"):
        os.makedirs(f"{outdir}/models")

    print("------------------------------------------------------------")
    print("Training V2A")
    print(f"Policy: {args.algo}")
    print(f"Env: {args.env}")
    print(f"Source type: {args.srctype}")
    print(f"Seed: {args.seed}")
    print(f"Source dataset: {src_dataset_path}")
    print(f"Z path: {z_path}")
    print(f"Source Q_z path: {src_Q_z_path}")
    print(f"Source V_z path: {src_V_z_path}")
    print(f"Output dir: {outdir}")
    print(f"Device: {device}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # Load source dataset and z
    # ------------------------------------------------------------
    src_dataset = load_hdf5_dataset(src_dataset_path)

    z_data = np.load(z_path)
    zs = z_data["zs"].astype(np.float32)

    assert zs.shape[0] == src_dataset["observations"].shape[0], (
        f"z size mismatch: zs has {zs.shape[0]} transitions, "
        f"source dataset has {src_dataset['observations'].shape[0]} transitions."
    )

    z_dim = zs.shape[1]

    # ------------------------------------------------------------
    # Load target dataset
    # ------------------------------------------------------------
    tar_env = gym.make(task + "-" + args.srctype + "-v2")
    tar_env.action_space.seed(args.seed)

    tar_dataset = d4rl.qlearning_dataset(tar_env)

    if args.limited_size:
        size = 5000
    else:
        size = int(tar_dataset["observations"].shape[0] * args.target_ratio)

    ind = np.random.randint(0, tar_dataset["observations"].shape[0], size=size)

    tar_dataset = {
        "observations": tar_dataset["observations"][ind],
        "actions": tar_dataset["actions"][ind],
        "next_observations": tar_dataset["next_observations"][ind],
        "rewards": tar_dataset["rewards"][ind],
        "terminals": tar_dataset["terminals"][ind],
    }

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------
    config = load_v2a_config(args, task)
    config = override_config(config, args)

    state_dim = tar_env.observation_space.shape[0]
    action_dim = tar_env.action_space.shape[0]
    max_action = float(tar_env.action_space.high[0])

    config.update({
        "env_name": args.env + args.srctype,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "tar_env_interact_interval": int(args.tar_env_interact_interval),
        "max_step": int(args.max_step),
        "tradeoff": args.tradeoff,
        "xi": args.xi,
        "z_dim": z_dim,
        "src_Q_z_path": src_Q_z_path,
        "src_V_z_path": src_V_z_path,
    })

    print("------------------------------------------------------------")
    print("Config:")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("------------------------------------------------------------")

    with open(outdir + "/log.txt", "w") as f:
        f.write("\n Policy: {}; Env: {}, seed: {}".format(
            args.algo,
            args.env,
            args.seed,
        ))
        f.write(f"\n source dataset: {src_dataset_path}")
        f.write(f"\n z path: {z_path}")
        f.write(f"\n src_Q_z_path: {src_Q_z_path}")
        f.write(f"\n src_V_z_path: {src_V_z_path}")

        for item in config.items():
            f.write("\n {}".format(item))

    # ------------------------------------------------------------
    # Build replay buffers
    # ------------------------------------------------------------
    src_replay_buffer = utils.SourceTrajectoryBuffer(
        state_dim,
        action_dim,
        device,
        max_size=int(1e6),
        max_episode_steps=1000,
    )

    tar_replay_buffer = utils.ReplayBuffer(
        state_dim,
        action_dim,
        device,
    )

    src_replay_buffer.convert_D4RL(src_dataset)
    src_replay_buffer.set_zs(zs)

    tar_replay_buffer.convert_D4RL(tar_dataset)

    print("Source trajectory buffer summary:")
    print(src_replay_buffer.summary())
    print(f"Loaded z shape: {zs.shape}")
    print(f"z std mean: {np.std(zs, axis=0).mean():.6f}")

    # ------------------------------------------------------------
    # Build policy
    # ------------------------------------------------------------
    policy = call_algo(args.algo, config, 3, device)

    # ------------------------------------------------------------
    # Initial eval
    # ------------------------------------------------------------
    eval_cnt = 0
    eval_tar_return = eval_policy(policy, tar_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    # ------------------------------------------------------------
    # Train dynamics alignment h(s,a,s')
    # ------------------------------------------------------------
    policy.update_info(
        src_replay_buffer,
        tar_replay_buffer,
        config["batch_size"],
        writer=None,
    )

    # ------------------------------------------------------------
    # Final V2A training
    # ------------------------------------------------------------
    for t in range(int(config["max_step"])):
        info = policy.train(
            src_replay_buffer,
            tar_replay_buffer,
            config["batch_size"],
            writer=None,
        )

        if (t + 1) % config["eval_freq"] == 0:
            tar_eval_return = eval_policy(
                policy,
                tar_env,
                eval_cnt=eval_cnt,
            )

            eval_normalized_score = tar_env.get_normalized_score(tar_eval_return)

            print(
                f"Step: {t + 1} "
                f"Score: {eval_normalized_score} "
                f"v_loss: {info['v_loss'].item():.6f} "
                f"q_loss: {info['q_loss'].item():.6f} "
                f"pi_loss: {info['pi_loss'].item():.6f} "
                f"num_select: {info['num_select']} "
                f"src_info_mean: {info['src_info_mean'].item():.6f} "
                f"src_adv_mean: {info['src_adv_mean'].item():.6f}"
            )

            with open(outdir + "/return.txt", "a") as f:
                f.write(f"{t + 1} {eval_normalized_score}\n")

            eval_cnt += 1

            if args.save_model:
                policy.save(f"{outdir}/models/model")

    if args.save_model:
        policy.save(f"{outdir}/models/model_final")