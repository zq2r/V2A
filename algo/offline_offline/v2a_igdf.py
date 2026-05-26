import copy
import math
from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from algo.offline.sql_z import DoubleQFuncZ, ValueFuncZ


class LinearEnsemble(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 3,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty((ensemble_size, in_features, out_features), **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((ensemble_size, 1, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -std, std)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        elif len(input.shape) > 3:
            raise ValueError("LinearEnsemble does not support inputs with >3 dims.")

        return torch.baddbmm(self.bias, input, self.weight)


class LayerNormEnsemble(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        ensemble_size: int = 3,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.ensemble_size = ensemble_size

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty((ensemble_size, 1, normalized_shape), **factory_kwargs)
            )
            self.bias = nn.Parameter(
                torch.empty((ensemble_size, 1, normalized_shape), **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        elif len(x.shape) > 3:
            raise ValueError("LayerNormEnsemble does not support inputs with >3 dims.")

        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)

        if self.elementwise_affine:
            x = x * self.weight + self.bias

        return x


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int = 3,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()

        assert normalization is None or normalization is LayerNormEnsemble

        net = []
        last_dim = input_dim

        for dim in hidden_layers:
            net.append(
                LinearEnsemble(
                    last_dim,
                    dim,
                    ensemble_size=ensemble_size,
                )
            )

            if dropout > 0.0:
                net.append(nn.Dropout(dropout))

            if normalization is not None:
                net.append(normalization(dim, ensemble_size=ensemble_size))

            net.append(act())
            last_dim = dim

        net.append(
            LinearEnsemble(
                last_dim,
                output_dim,
                ensemble_size=ensemble_size,
            )
        )

        if output_act is not None:
            net.append(output_act())

        self.net = nn.Sequential(*net)
        self._has_output_act = output_act is not None

    def forward(self, x):
        return self.net(x)

    @property
    def last_layer(self):
        if self._has_output_act:
            return self.net[-2]
        return self.net[-1]


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)

    @property
    def last_layer(self):
        return self.network[-1]


class ContrastiveInfo(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        repr_dim: int,
        ensemble_size: int = 1,
        repr_norm: bool = False,
        repr_norm_temp: bool = False,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.repr_dim = repr_dim
        self.ensemble_size = ensemble_size
        self.repr_norm = repr_norm
        self.repr_norm_temp = repr_norm_temp

        input_dim_for_sa = state_dim + action_dim
        input_dim_for_ss = state_dim

        if self.ensemble_size > 1:
            self.encoder_sa = EnsembleMLP(
                input_dim_for_sa,
                repr_dim,
                ensemble_size=ensemble_size,
                **kwargs,
            )
            self.encoder_ss = EnsembleMLP(
                input_dim_for_ss,
                repr_dim,
                ensemble_size=ensemble_size,
                **kwargs,
            )
        else:
            self.encoder_sa = MLPNetwork(
                input_dim_for_sa,
                repr_dim,
                **kwargs,
            )
            self.encoder_ss = MLPNetwork(
                input_dim_for_ss,
                repr_dim,
                **kwargs,
            )

        self.ortho_init = ortho_init
        self.output_gain = output_gain

    def encode(self, obs, action, ss):
        sa_repr = self.encoder_sa(torch.cat([obs, action], dim=-1))
        ss_repr = self.encoder_ss(ss)

        if self.repr_norm:
            sa_repr = sa_repr / (
                torch.linalg.norm(sa_repr, dim=-1, keepdim=True) + 1e-8
            )
            ss_repr = ss_repr / (
                torch.linalg.norm(ss_repr, dim=-1, keepdim=True) + 1e-8
            )

        if self.repr_norm_temp:
            raise NotImplementedError("Running normalization is not implemented.")

        return sa_repr, ss_repr

    def combine_repr(self, sa_repr, ss_repr):
        if len(sa_repr.shape) == 2 and len(ss_repr.shape) == 2:
            return torch.einsum("iz,jz->ij", sa_repr, ss_repr)

        return torch.einsum("eiz,ejz->eij", sa_repr, ss_repr)

    def forward(self, obs, action, ss, return_repr=False):
        sa_repr, ss_repr = self.encode(obs, action, ss)
        logits = self.combine_repr(sa_repr, ss_repr)

        if return_repr:
            return logits, sa_repr, ss_repr

        return logits


class TanhTransform(Transform):
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()

        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)

        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()

        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)

        action = dist.rsample()

        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None

        mean = torch.tanh(mu)

        return action * self.max_action, logprob, mean * self.max_action

    def bc_loss(self, state, action):
        mu_logstd = self.network(state)
        mu, _ = mu_logstd.chunk(2, dim=1)
        pred_action = torch.tanh(mu)
        return (pred_action - action) ** 2


class DoubleQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()

        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class ValueFunc(nn.Module):
    def __init__(self, state_dim, action_dim=None, hidden_size=256):
        super(ValueFunc, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class V2A_IGDF(object):
    """
    Standalone V2A final policy learning.

    It uses:
        h(s,a,s') from ContrastiveInfo
        A(s,a,z) = Q_z(s,a,z) - V_z(s,z) from SQL_Z
        f(s,a,s',z) = tradeoff * h + (1 - tradeoff) * Norm(A)

    Final actor/critic/value are still IQL-style and do not condition on z.
    """

    def __init__(self, config, device, target_entropy=None):
        self.config = config
        self.device = device

        self.discount = config["gamma"]
        self.tau = config["tau"]
        self.target_entropy = target_entropy if target_entropy else -config["action_dim"]
        self.update_interval = config["update_interval"]

        # ------------------------------------------------------------
        # Source modality-aware Q/V:
        #     Q_src(s,a,z), V_src(s,z)
        # ------------------------------------------------------------
        self.src_Q = DoubleQFuncZ(
            config["state_dim"],
            config["action_dim"],
            config["z_dim"],
            hidden_size=config["hidden_sizes"],
        ).to(self.device)

        self.src_V = ValueFuncZ(
            config["state_dim"],
            config["z_dim"],
            hidden_size=config["hidden_sizes"],
        ).to(self.device)

        self.src_Q.load_state_dict(
            torch.load(config["src_Q_z_path"], map_location=self.device)
        )
        self.src_V.load_state_dict(
            torch.load(config["src_V_z_path"], map_location=self.device)
        )

        self.src_Q.eval()
        self.src_V.eval()

        for p in self.src_Q.parameters():
            p.requires_grad = False
        for p in self.src_V.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------
        # Final IQL modules
        # ------------------------------------------------------------
        self.lam = config["lam"]
        self.temp = config["temp"]
        self.total_it = 0

        self.q_funcs = DoubleQFunc(
            config["state_dim"],
            config["action_dim"],
            hidden_size=config["hidden_sizes"],
        ).to(self.device)

        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()

        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        self.v_func = ValueFunc(
            config["state_dim"],
            config["action_dim"],
            hidden_size=config["hidden_sizes"],
        ).to(self.device)

        self.policy = Policy(
            config["state_dim"],
            config["action_dim"],
            config["max_action"],
            hidden_size=config["hidden_sizes"],
        ).to(self.device)

        self.q_optimizer = torch.optim.Adam(
            self.q_funcs.parameters(),
            lr=config["critic_lr"],
        )
        self.v_optimizer = torch.optim.Adam(
            self.v_func.parameters(),
            lr=config["critic_lr"],
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config["actor_lr"],
        )
        self.policy_lr_schedule = CosineAnnealingLR(
            self.policy_optimizer,
            config["max_step"],
        )

        # ------------------------------------------------------------
        # Dynamics alignment h(s,a,s')
        # ------------------------------------------------------------
        self.info = ContrastiveInfo(
            config["state_dim"],
            config["action_dim"],
            config["repr_dim"],
            config["ensemble_size"],
            config["repr_norm"],
            config["repr_norm_temp"],
            config["ortho_init"],
            config["output_gain"],
        ).to(self.device)

        self.info_optimizer = torch.optim.Adam(
            self.info.parameters(),
            lr=config["actor_lr"],
        )

    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(
                torch.Tensor(state).view(1, -1).to(self.device)
            )

            if test:
                return mean.squeeze().cpu().numpy()
            return action.squeeze().cpu().numpy()

    def _reduce_update_info_logits(self, logits):
        """
        Convert ContrastiveInfo logits to [B, B] for BCE loss.
        """
        if logits.dim() == 4:
            logits = logits.squeeze(2).mean(dim=0)
        elif logits.dim() == 3:
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)
            else:
                logits = logits.mean(dim=0)

        return logits

    def update_info(self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None):
        info_step = 0

        for _ in trange(self.config["info_update_step"], desc="Training"):
            info_step += 1

            tar_s, tar_a, tar_ss, _, _ = tar_replay_buffer.sample(batch_size)
            _, _, src_ss, _, _ = src_replay_buffer.sample(batch_size - 1)

            tar_s = tar_s.unsqueeze(1)
            tar_a = tar_a.unsqueeze(1)
            tar_ss = tar_ss.unsqueeze(1)

            src_ss = src_ss.unsqueeze(0)
            src_ss = src_ss.expand(batch_size, -1, -1)

            ss = torch.cat((tar_ss, src_ss), dim=1)

            logits = self.info(tar_s, tar_a, ss)
            logits = self._reduce_update_info_logits(logits)

            matrix = torch.zeros(
                (batch_size, batch_size),
                dtype=torch.float32,
                device=self.device,
            )
            matrix[:, 0] = 1.0

            info_loss = F.binary_cross_entropy_with_logits(logits, matrix)
            info_loss = torch.mean(info_loss)

            if writer is not None and info_step % 100 == 0:
                writer.add_scalar("train/info loss", info_loss, info_step)

            self.info_optimizer.zero_grad()
            info_loss.backward()
            self.info_optimizer.step()

    def update_target(self):
        with torch.no_grad():
            for target_q_param, q_param in zip(
                self.target_q_funcs.parameters(),
                self.q_funcs.parameters(),
            ):
                target_q_param.data.copy_(
                    self.tau * q_param.data
                    + (1.0 - self.tau) * target_q_param.data
                )

    def update_v_function(self, state_batch, action_batch, writer=None):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)

        v = self.v_func(state_batch)
        adv = q_t - v

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar("train/adv", adv.mean(), self.total_it)
            writer.add_scalar("train/value", v.mean(), self.total_it)

        v_loss = asymmetric_l2_loss(adv, self.lam)

        return v_loss, adv

    def update_q_functions(
        self,
        state_batch,
        action_batch,
        reward_batch,
        nextstate_batch,
        not_done_batch,
        mask,
        writer=None,
    ):
        with torch.no_grad():
            v_t = self.v_func(nextstate_batch)
            value_target = reward_batch + not_done_batch * self.discount * v_t

        q_1, q_2 = self.q_funcs(state_batch, action_batch)

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar("train/q1", q_1.mean(), self.total_it)

        loss = (mask * (q_1 - value_target) ** 2).mean() + \
               (mask * (q_2 - value_target) ** 2).mean()

        return loss

    def update_policy(self, advantage_batch, state_batch, action_batch):
        exp_adv = torch.exp(self.temp * advantage_batch.detach()).clamp(max=100.0)
        bc_loss = self.policy.bc_loss(state_batch, action_batch)
        policy_loss = torch.mean(exp_adv * bc_loss)
        return policy_loss

    def _compute_src_info(self, src_state, src_action, src_next_state):
        """
        Compute h(s,a,s') for source samples.
        Return shape:
            [B, 1]
        """
        if self.config["repr_norm"]:
            logits = self.info(src_state, src_action, src_next_state)

            if logits.dim() == 3:
                diagonal = torch.diagonal(logits, dim1=-2, dim2=-1)
                src_info = diagonal.mean(dim=0).reshape(-1, 1)
            else:
                src_info = torch.diag(logits).reshape(-1, 1)

        else:
            logits, srcsa_repr, srcss_repr = self.info(
                src_state,
                src_action,
                src_next_state,
                return_repr=True,
            )

            if logits.dim() == 3:
                diagonal = torch.diagonal(logits, dim1=-2, dim2=-1)

                srcsa_norm = torch.linalg.norm(srcsa_repr, dim=-1)
                srcss_norm = torch.linalg.norm(srcss_repr, dim=-1)

                src_info = diagonal / (srcsa_norm * srcss_norm + 1e-8)
                src_info = src_info.mean(dim=0).reshape(-1, 1)
            else:
                diagonal = torch.diag(logits).reshape(-1, 1)

                srcsa_norm = torch.linalg.norm(
                    srcsa_repr,
                    dim=-1,
                    keepdim=True,
                )
                srcss_norm = torch.linalg.norm(
                    srcss_repr,
                    dim=-1,
                    keepdim=True,
                )

                src_info = diagonal / (srcsa_norm * srcss_norm + 1e-8)

        return src_info

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        self.total_it += 1

        (
            src_state,
            src_action,
            src_next_state,
            src_reward,
            src_not_done,
            src_z,
        ) = src_replay_buffer.sample(batch_size, with_z=True)

        (
            tar_state,
            tar_action,
            tar_next_state,
            tar_reward,
            tar_not_done,
        ) = tar_replay_buffer.sample(batch_size)

        # ------------------------------------------------------------
        # V2A filtering:
        #     f(s,a,s',z) = λ h(s,a,s') + (1-λ) Norm(A(s,a,z))
        # ------------------------------------------------------------
        with torch.no_grad():
            src_info = self._compute_src_info(
                src_state,
                src_action,
                src_next_state,
            )

            src_q1, src_q2 = self.src_Q(src_state, src_action, src_z)
            src_q = torch.min(src_q1, src_q2)

            src_v = self.src_V(src_state, src_z)
            src_adv = src_q - src_v

            eps = 1e-8
            min_val = torch.min(src_adv, dim=0, keepdim=True).values
            max_val = torch.max(src_adv, dim=0, keepdim=True).values
            src_adv_norm = (src_adv - min_val) / (max_val - min_val + eps)

            filter_info = (
                self.config["tradeoff"] * src_info
                + (1.0 - self.config["tradeoff"]) * src_adv_norm
            )

            num_select = int(batch_size * float(self.config["xi"]))
            num_select = max(1, min(batch_size, num_select))

            sorted_indices = torch.argsort(filter_info[:, 0])
            top_indices = sorted_indices[-num_select:]

            info_temp = torch.exp(src_info[top_indices]).clamp(max=100.0)

        src_state = src_state[top_indices]
        src_action = src_action[top_indices]
        src_next_state = src_next_state[top_indices]
        src_reward = src_reward[top_indices]
        src_not_done = src_not_done[top_indices]

        mask = torch.ones((num_select + batch_size, 1), device=self.device)
        mask[:num_select] = info_temp

        state = torch.cat([src_state, tar_state], dim=0)
        action = torch.cat([src_action, tar_action], dim=0)
        next_state = torch.cat([src_next_state, tar_next_state], dim=0)
        reward = torch.cat([src_reward, tar_reward], dim=0)
        not_done = torch.cat([src_not_done, tar_not_done], dim=0)

        v_loss_step, adv = self.update_v_function(state, action, writer)

        self.v_optimizer.zero_grad()
        v_loss_step.backward()
        self.v_optimizer.step()

        q_loss_step = self.update_q_functions(
            state,
            action,
            reward,
            next_state,
            not_done,
            mask,
            writer,
        )

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        for p in self.q_funcs.parameters():
            p.requires_grad = False

        pi_loss_step = self.update_policy(adv, state, action)

        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

        return {
            "v_loss": v_loss_step.detach(),
            "q_loss": q_loss_step.detach(),
            "pi_loss": pi_loss_step.detach(),
            "num_select": num_select,
            "src_info_mean": src_info.mean().detach(),
            "src_adv_mean": src_adv.mean().detach(),
            "src_adv_norm_mean": src_adv_norm.mean().detach(),
            "filter_info_mean": filter_info.mean().detach(),
        }

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.v_func.state_dict(), filename + "_value")
        torch.save(self.v_optimizer.state_dict(), filename + "_value_optimizer")

        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(
            self.policy_lr_schedule.state_dict(),
            filename + "_actor_lr_scheduler",
        )

    def load(self, filename):
        self.q_funcs.load_state_dict(
            torch.load(filename + "_critic", map_location=self.device)
        )
        self.q_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=self.device)
        )

        self.v_func.load_state_dict(
            torch.load(filename + "_value", map_location=self.device)
        )
        self.v_optimizer.load_state_dict(
            torch.load(filename + "_value_optimizer", map_location=self.device)
        )

        self.policy.load_state_dict(
            torch.load(filename + "_actor", map_location=self.device)
        )
        self.policy_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", map_location=self.device)
        )
        self.policy_lr_schedule.load_state_dict(
            torch.load(filename + "_actor_lr_scheduler", map_location=self.device)
        )