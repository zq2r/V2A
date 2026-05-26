import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DoubleQFuncZ(nn.Module):
    """
    Modality-aware double Q-function:
        Q(s, a, z)

    Input:
        state:  [B, state_dim]
        action: [B, action_dim]
        z:      [B, z_dim]
    """

    def __init__(self, state_dim, action_dim, z_dim, hidden_size=256):
        super(DoubleQFuncZ, self).__init__()

        input_dim = state_dim + action_dim + z_dim

        self.network1 = MLPNetwork(input_dim, 1, hidden_size)
        self.network2 = MLPNetwork(input_dim, 1, hidden_size)

    def forward(self, state, action, z):
        x = torch.cat((state, action, z), dim=1)
        return self.network1(x), self.network2(x)


class ValueFuncZ(nn.Module):
    """
    Modality-aware value function:
        V(s, z)

    Input:
        state: [B, state_dim]
        z:     [B, z_dim]
    """

    def __init__(self, state_dim, z_dim, hidden_size=256):
        super(ValueFuncZ, self).__init__()

        input_dim = state_dim + z_dim
        self.network = MLPNetwork(input_dim, 1, hidden_size)

    def forward(self, state, z):
        x = torch.cat((state, z), dim=1)
        return self.network(x)


def sql_asymmetric_l2_loss(v: torch.Tensor, adv: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Sparse-QL value loss used by the original SQL implementation.

    Here:
        adv = Q(s,a,z) - V(s,z)
    """
    sp_term = adv / (2 * alpha) + 1.0
    sp_weight = torch.where(sp_term > 0, 1.0, 0.0)
    value_loss = torch.mean(sp_weight * (sp_term ** 2) + v / alpha)
    return value_loss


class SQL_Z(object):
    """
    Modality-aware Sparse-QL for V2A.

    This class only pretrains Q(s,a,z) and V(s,z).
    It does not train a policy, because V2A only needs:
        A(s,a,z) = Q(s,a,z) - V(s,z)

    The interface follows DVDF's SQL/IQL style:
        policy.train(buffer, batch_size)
        policy.save(path)
        policy.load(path)
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.discount = config["gamma"]
        self.tau = config["tau"]
        self.update_interval = config.get("update_interval", 1)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.z_dim = config["z_dim"]
        self.hidden_size = config["hidden_sizes"]

        # Avoid using self.alpha because the original SQL class has an
        # alpha property issue. Use sql_alpha explicitly.
        self.sql_alpha = config["alpha"]

        self.total_it = 0

        self.q_funcs = DoubleQFuncZ(
            self.state_dim,
            self.action_dim,
            self.z_dim,
            hidden_size=self.hidden_size,
        ).to(self.device)

        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()

        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        self.v_func = ValueFuncZ(
            self.state_dim,
            self.z_dim,
            hidden_size=self.hidden_size,
        ).to(self.device)

        self.q_optimizer = torch.optim.Adam(
            self.q_funcs.parameters(),
            lr=config["critic_lr"],
        )

        self.v_optimizer = torch.optim.Adam(
            self.v_func.parameters(),
            lr=config["critic_lr"],
        )

    def update_target(self):
        """
        Moving average update of target Q networks.
        """
        with torch.no_grad():
            for target_q_param, q_param in zip(
                self.target_q_funcs.parameters(),
                self.q_funcs.parameters(),
            ):
                target_q_param.data.copy_(
                    self.tau * q_param.data
                    + (1.0 - self.tau) * target_q_param.data
                )

    def update_v_function(self, state_batch, action_batch, z_batch, writer=None):
        """
        Update V(s,z) with modality-aware Sparse-QL value objective.
        """
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(
                state_batch,
                action_batch,
                z_batch,
            )
            q_t = torch.min(q_t1, q_t2)

        v = self.v_func(state_batch, z_batch)
        adv = q_t - v

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar("train_z/adv", adv.mean(), self.total_it)
            writer.add_scalar("train_z/value", v.mean(), self.total_it)

        v_loss = sql_asymmetric_l2_loss(v, adv, self.sql_alpha)

        return v_loss, adv

    def update_q_functions(
        self,
        state_batch,
        action_batch,
        reward_batch,
        nextstate_batch,
        not_done_batch,
        z_batch,
        writer=None,
    ):
        """
        Update Q(s,a,z) with Bellman target:
            r + gamma * V(s', z)

        The same trajectory-level z is used for s and s'.
        """
        with torch.no_grad():
            v_t = self.v_func(nextstate_batch, z_batch)
            value_target = reward_batch + not_done_batch * self.discount * v_t

        q_1, q_2 = self.q_funcs(state_batch, action_batch, z_batch)

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar("train_z/q1", q_1.mean(), self.total_it)
            writer.add_scalar("train_z/q2", q_2.mean(), self.total_it)

        q_loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)

        return q_loss

    def train(self, src_replay_buffer, batch_size=128, writer=None):
        """
        One SQL_Z update.

        src_replay_buffer must be SourceTrajectoryBuffer with z already set:
            src_replay_buffer.set_zs(zs)
        """
        self.total_it += 1

        (
            state,
            action,
            next_state,
            reward,
            not_done,
            z,
        ) = src_replay_buffer.sample(batch_size, with_z=True)

        v_loss_step, adv = self.update_v_function(
            state,
            action,
            z,
            writer,
        )

        self.v_optimizer.zero_grad()
        v_loss_step.backward()
        self.v_optimizer.step()

        q_loss_step = self.update_q_functions(
            state,
            action,
            reward,
            next_state,
            not_done,
            z,
            writer,
        )

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        if self.total_it % self.update_interval == 0:
            self.update_target()

        info = {
            "v_loss": v_loss_step.detach(),
            "q_loss": q_loss_step.detach(),
            "adv_mean": adv.mean().detach(),
            "adv_std": adv.std().detach(),
        }

        return info

    @torch.no_grad()
    def get_advantage(self, state, action, z):
        """
        Compute A(s,a,z) = min(Q1,Q2)(s,a,z) - V(s,z).
        Used later by V2A filtering.
        """
        q1, q2 = self.q_funcs(state, action, z)
        q = torch.min(q1, q2)
        v = self.v_func(state, z)
        adv = q - v
        return adv

    def save(self, filename):
        """
        DVDF-style save.

        If filename = '.../models/model', this creates:
            .../models/model_critic
            .../models/model_critic_optimizer
            .../models/model_value
            .../models/model_value_optimizer
        """
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.v_func.state_dict(), filename + "_value")
        torch.save(self.v_optimizer.state_dict(), filename + "_value_optimizer")

    def load(self, filename, load_optimizer=True):
        self.q_funcs.load_state_dict(
            torch.load(filename + "_critic", map_location=self.device)
        )
        self.v_func.load_state_dict(
            torch.load(filename + "_value", map_location=self.device)
        )

        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        if load_optimizer:
            self.q_optimizer.load_state_dict(
                torch.load(filename + "_critic_optimizer", map_location=self.device)
            )
            self.v_optimizer.load_state_dict(
                torch.load(filename + "_value_optimizer", map_location=self.device)
            )