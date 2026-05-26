import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional

from torch.nn.modules.dropout import Dropout
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]
        

class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activations: Callable = nn.ReLU,
        activate_final: int = False,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()

        self.affines = []
        self.affines.append(nn.Linear(in_dim, hidden_dim))
        for i in range(n_layers-2):
            self.affines.append(nn.Linear(hidden_dim, hidden_dim))
        self.affines.append(nn.Linear(hidden_dim, out_dim))
        self.affines = nn.ModuleList(self.affines)

        self.activations = activations()
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)
            self.norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for i in range(len(self.affines)):
            x = self.affines[i](x)
            if i != len(self.affines)-1 or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)
                    # x = self.norm_layer(x)
        return x

def identity(x):
    return x

def fanin_init(tensor, scale=1):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = scale / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def orthogonal_init(tensor, gain=0.01):
    torch.nn.init.orthogonal_(tensor, gain=gain)

class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            final_init_scale=None,
            dropout_rate=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.fcs = []

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        state_dim = inputs[0].shape[-1]
        
        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)
        
        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            # add dropout
            if self.dropout_rate:
                h = self.dropout(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output
    
    def sample(self, *inputs):
        preds = self.forward(*inputs)

        sample_idxs = np.random.choice(self.ensemble_size, 2, replace=False)
        preds_sample = preds[sample_idxs]
        
        return torch.min(preds_sample, dim=0)[0], sample_idxs
    
# 加入sequence buffer
def _to_numpy(x):
    """
    Convert h5py dataset / numpy array / torch tensor to numpy array.
    This is useful because source datasets may come from h5py.File.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_dataset_field(dataset, key, default=None):
    """
    Read one field from dict-like or h5py-like dataset.
    """
    if key in dataset:
        return _to_numpy(dataset[key])
    return default


def _require_dataset_field(dataset, key):
    value = _get_dataset_field(dataset, key, default=None)
    if value is None:
        raise KeyError(f"Dataset does not contain required field: {key}")
    return value


def split_into_trajectories(terminals, timeouts=None, max_episode_steps=None):
    """
    Split a transition-level dataset into trajectories.

    terminals[i] or timeouts[i] means transition i is the last transition
    of the current trajectory, so transition i is included in that trajectory.

    Args:
        terminals: shape [N] or [N, 1]
        timeouts: optional, shape [N] or [N, 1]
        max_episode_steps: optional. If terminals/timeouts are missing or unreliable,
            this can force a cut every fixed horizon, e.g., 1000 for MuJoCo.

    Returns:
        trajectories: list of np.ndarray. Each element contains transition indices
            belonging to one trajectory.
    """
    terminals = np.asarray(terminals).reshape(-1).astype(bool)
    n = len(terminals)

    if timeouts is None:
        timeouts = np.zeros(n, dtype=bool)
    else:
        timeouts = np.asarray(timeouts).reshape(-1).astype(bool)

    trajectories = []
    start = 0

    for i in range(n):
        episode_len = i - start + 1

        end_by_terminal = terminals[i]
        end_by_timeout = timeouts[i]
        end_by_max_len = (
            max_episode_steps is not None and episode_len >= max_episode_steps
        )

        if end_by_terminal or end_by_timeout or end_by_max_len:
            if i + 1 > start:
                trajectories.append(np.arange(start, i + 1, dtype=np.int64))
            start = i + 1

    if start < n:
        trajectories.append(np.arange(start, n, dtype=np.int64))

    return trajectories


class SourceTrajectoryBuffer(object):
    """
    Source dataset buffer for V2A.

    This class keeps the source dataset in transition-level format, but also
    builds trajectory indices for temporally-consistent representation learning.

    It supports:
        1. sample_trajectories(): for q_psi(z | tau) and p_theta(s' | s, a, z)
        2. sample(): transition-level sampling, compatible with later SQL_Z/V2A
        3. set_zs(): attach inferred trajectory-level z to every transition

    Important:
        - The original ReplayBuffer is kept unchanged.
        - This class should be used only in V2A-related scripts.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=int(1e6),
        max_episode_steps=None,
    ):
        self.max_size = max_size
        self.device = device
        self.max_episode_steps = max_episode_steps

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.terminals = np.zeros((max_size, 1), dtype=np.float32)
        self.timeouts = np.zeros((max_size, 1), dtype=np.float32)

        self.size = 0

        self.traj_indices = None
        self.traj_lengths = None
        self.num_trajectories = 0

        # Will be set after representation learning.
        self.z = None
        self.z_dim = None

    def convert_D4RL(self, dataset):
        """
        Convert D4RL/hdf5-style dataset and build trajectory index list.

        Expected fields:
            observations
            actions
            next_observations
            rewards
            terminals

        Optional fields:
            timeouts
        """
        observations = _require_dataset_field(dataset, "observations").astype(np.float32)
        actions = _require_dataset_field(dataset, "actions").astype(np.float32)
        next_observations = _require_dataset_field(dataset, "next_observations").astype(np.float32)
        rewards = _require_dataset_field(dataset, "rewards").astype(np.float32).reshape(-1, 1)

        terminals = _get_dataset_field(dataset, "terminals", default=None)
        if terminals is None:
            terminals = _get_dataset_field(dataset, "dones", default=None)
        if terminals is None:
            raise KeyError("Dataset must contain `terminals` or `dones`.")

        terminals = terminals.astype(np.float32).reshape(-1, 1)

        timeouts = _get_dataset_field(dataset, "timeouts", default=None)
        if timeouts is None:
            timeouts = np.zeros_like(terminals, dtype=np.float32)
        else:
            timeouts = timeouts.astype(np.float32).reshape(-1, 1)

        size = observations.shape[0]
        assert size <= self.max_size, f"Dataset size {size} exceeds max_size {self.max_size}."
        assert actions.shape[0] == size
        assert next_observations.shape[0] == size
        assert rewards.shape[0] == size
        assert terminals.shape[0] == size
        assert timeouts.shape[0] == size

        self.state = observations
        self.action = actions
        self.next_state = next_observations
        self.reward = rewards

        # For bootstrapping, timeout is usually not treated as true terminal.
        self.not_done = 1.0 - terminals

        self.terminals = terminals
        self.timeouts = timeouts
        self.size = size

        self.traj_indices = split_into_trajectories(
            terminals=self.terminals,
            timeouts=self.timeouts,
            max_episode_steps=self.max_episode_steps,
        )
        self.num_trajectories = len(self.traj_indices)
        self.traj_lengths = np.array(
            [len(idx) for idx in self.traj_indices],
            dtype=np.int64,
        )

    def summary(self):
        if self.traj_lengths is None or len(self.traj_lengths) == 0:
            return {
                "num_transitions": int(self.size),
                "num_trajectories": 0,
                "state_dim": int(self.state_dim),
                "action_dim": int(self.action_dim),
            }

        return {
            "num_transitions": int(self.size),
            "num_trajectories": int(self.num_trajectories),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "min_traj_len": int(self.traj_lengths.min()),
            "max_traj_len": int(self.traj_lengths.max()),
            "mean_traj_len": float(self.traj_lengths.mean()),
        }

    def set_zs(self, z):
        """
        Attach inferred z to each transition.

        Args:
            z: np.ndarray or torch.Tensor with shape [num_transitions, z_dim]
        """
        z = _to_numpy(z).astype(np.float32)
        assert z.ndim == 2
        assert z.shape[0] == self.size

        self.z = z
        self.z_dim = z.shape[-1]

    def sample(self, batch_size, with_z=False):
        """
        Transition-level sampling.

        If with_z=False:
            return state, action, next_state, reward, not_done

        If with_z=True:
            return state, action, next_state, reward, not_done, z
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        state = torch.FloatTensor(self.state[ind]).to(self.device)
        action = torch.FloatTensor(self.action[ind]).to(self.device)
        next_state = torch.FloatTensor(self.next_state[ind]).to(self.device)
        reward = torch.FloatTensor(self.reward[ind]).to(self.device)
        not_done = torch.FloatTensor(self.not_done[ind]).to(self.device)

        if with_z:
            if self.z is None:
                raise RuntimeError("with_z=True, but z has not been set. Call set_zs() first.")
            z = torch.FloatTensor(self.z[ind]).to(self.device)
            return state, action, next_state, reward, not_done, z

        return state, action, next_state, reward, not_done

    def sample_trajectories(self, batch_size, max_len=None):
        """
        Sample a padded batch of trajectories.

        Args:
            batch_size: number of trajectories.
            max_len: if not None, long trajectories will be randomly cropped.

        Returns:
            batch dict:
                observations:      [B, T, state_dim]
                actions:           [B, T, action_dim]
                next_observations: [B, T, state_dim]
                rewards:           [B, T, 1]
                not_dones:         [B, T, 1]
                mask:              [B, T, 1]
                indices:           [B, T], original transition indices, -1 for padding
                lengths:           [B]
                traj_ids:          [B]
        """
        if self.traj_indices is None or self.num_trajectories == 0:
            raise RuntimeError("Trajectory indices are not built. Call convert_D4RL() first.")

        traj_ids = np.random.randint(0, self.num_trajectories, size=batch_size)
        selected_indices = []

        for traj_id in traj_ids:
            idx = self.traj_indices[traj_id]
            traj_len = len(idx)

            if max_len is not None and traj_len > max_len:
                start = np.random.randint(0, traj_len - max_len + 1)
                idx = idx[start:start + max_len]

            selected_indices.append(idx)

        lengths = np.array([len(idx) for idx in selected_indices], dtype=np.int64)
        t_max = int(lengths.max())

        obs_batch = np.zeros((batch_size, t_max, self.state_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, t_max, self.action_dim), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, t_max, self.state_dim), dtype=np.float32)
        rew_batch = np.zeros((batch_size, t_max, 1), dtype=np.float32)
        not_done_batch = np.zeros((batch_size, t_max, 1), dtype=np.float32)
        mask_batch = np.zeros((batch_size, t_max, 1), dtype=np.float32)
        index_batch = -np.ones((batch_size, t_max), dtype=np.int64)

        for b, idx in enumerate(selected_indices):
            t = len(idx)

            obs_batch[b, :t] = self.state[idx]
            act_batch[b, :t] = self.action[idx]
            next_obs_batch[b, :t] = self.next_state[idx]
            rew_batch[b, :t] = self.reward[idx]
            not_done_batch[b, :t] = self.not_done[idx]
            mask_batch[b, :t] = 1.0
            index_batch[b, :t] = idx

        return {
            "observations": torch.FloatTensor(obs_batch).to(self.device),
            "actions": torch.FloatTensor(act_batch).to(self.device),
            "next_observations": torch.FloatTensor(next_obs_batch).to(self.device),
            "rewards": torch.FloatTensor(rew_batch).to(self.device),
            "not_dones": torch.FloatTensor(not_done_batch).to(self.device),
            "mask": torch.FloatTensor(mask_batch).to(self.device),
            "indices": torch.LongTensor(index_batch).to(self.device),
            "lengths": torch.LongTensor(lengths).to(self.device),
            "traj_ids": torch.LongTensor(traj_ids).to(self.device),
        }

    def iter_trajectories(self):
        """
        Iterate over all trajectories.

        This will be used later for relabeling:
            trajectory -> encoder -> z
            assign this z to all transition indices in the trajectory.
        """
        if self.traj_indices is None:
            raise RuntimeError("Trajectory indices are not built. Call convert_D4RL() first.")

        for traj_id, idx in enumerate(self.traj_indices):
            yield {
                "traj_id": traj_id,
                "indices": idx,
                "observations": self.state[idx],
                "actions": self.action[idx],
                "next_observations": self.next_state[idx],
                "rewards": self.reward[idx],
                "not_dones": self.not_done[idx],
                "terminals": self.terminals[idx],
                "timeouts": self.timeouts[idx],
            }