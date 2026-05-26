import torch
import torch.nn as nn

import algo.utils as utils


class TrajectoryEncoder(nn.Module):
    """
    Trajectory-level encoder q_psi(z | tau).

    Input:
        observations:      [B, T, state_dim]
        actions:           [B, T, action_dim]
        next_observations: [B, T, state_dim]
        mask:              [B, T, 1]

    Output:
        mean:    [B, z_dim]
        log_std: [B, z_dim]

    We use x_t = concat(s_t, a_t, delta_s_t), where
        delta_s_t = s_{t+1} - s_t.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        z_dim=8,
        hidden_size=256,
        num_layers=1,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super(TrajectoryEncoder, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size

        input_dim = state_dim + action_dim + state_dim

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mean_layer = nn.Linear(hidden_size, z_dim)
        self.log_std_layer = nn.Linear(hidden_size, z_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, observations, actions, next_observations, mask):
        delta = next_observations - observations
        x = torch.cat([observations, actions, delta], dim=-1)

        # out: [B, T, hidden_size]
        out, _ = self.rnn(x)

        # Select the last valid hidden state according to mask.
        lengths = mask.squeeze(-1).sum(dim=1).long().clamp(min=1)
        last_index = lengths - 1

        batch_index = torch.arange(out.shape[0], device=out.device)
        h = out[batch_index, last_index]

        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, observations, actions, next_observations, mask):
        mean, log_std = self.forward(
            observations,
            actions,
            next_observations,
            mask,
        )

        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z, mean, log_std

    @torch.no_grad()
    def infer_mean(self, observations, actions, next_observations, mask):
        mean, _ = self.forward(
            observations,
            actions,
            next_observations,
            mask,
        )
        return mean


class EnsembleConditionalDynamicsModel(nn.Module):
    """
    Ensemble dynamics decoder p_theta(s' | s, a, z).

    This follows the existing DVDF utility implementation:
        utils.ParallelizedEnsembleFlattenMLP

    Input:
        state:  [N, state_dim]
        action: [N, action_dim]
        z:      [N, z_dim]

    Output:
        pred_delta: [ensemble_size, N, state_dim]

    We predict delta_s = s' - s instead of directly predicting s'.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        z_dim=8,
        hidden_size=256,
        n_layers=3,
        ensemble_size=3,
    ):
        super(EnsembleConditionalDynamicsModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.ensemble_size = ensemble_size

        input_size = state_dim + action_dim + z_dim
        output_size = state_dim

        # n_layers means total number of linear layers.
        # ParallelizedEnsembleFlattenMLP takes hidden_sizes and creates
        # one extra final layer internally.
        hidden_sizes = [hidden_size for _ in range(n_layers - 1)]

        self.network = utils.ParallelizedEnsembleFlattenMLP(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=input_size,
            output_size=output_size,
        )

    def forward(self, state, action, z):
        pred_delta = self.network(state, action, z)
        return pred_delta


def gaussian_kl_to_standard_normal(mean, log_std):
    """
    Compute KL( N(mean, std^2) || N(0, I) ).

    Args:
        mean:    [B, z_dim]
        log_std: [B, z_dim]

    Returns:
        scalar tensor.
    """
    var = torch.exp(2.0 * log_std)
    kl = -0.5 * (1.0 + 2.0 * log_std - mean.pow(2) - var)
    return kl.sum(dim=-1).mean()


class TCELBO(object):
    """
    Temporally-consistent modality representation learning.

    This class follows the code style of DVDF/IQL/SQL:
        - owns networks and optimizers
        - provides train(), save(), load()
        - performs one update per train() call

    E-step:
        Fix decoder p_theta, update trajectory encoder q_psi(z | tau).

    M-step:
        Fix encoder q_psi, update dynamics decoder p_theta(s' | s, a, z).

    Practical likelihood:
        We use masked MSE for delta_s prediction as the negative log-likelihood
        surrogate.
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]

        self.z_dim = config.get("z_dim", 8)
        self.hidden_size = config.get("hidden_sizes", 256)
        self.n_layers = config.get("n_layers", 3)
        self.rnn_layers = config.get("rnn_layers", 1)
        self.ensemble_size = config.get("ensemble_size", 3)

        self.beta_kl = config.get("beta_kl", 1e-3)
        self.grad_clip_norm = config.get("grad_clip_norm", 10.0)

        self.encoder = TrajectoryEncoder(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            z_dim=self.z_dim,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            log_std_min=config.get("log_std_min", -5.0),
            log_std_max=config.get("log_std_max", 2.0),
        ).to(self.device)

        self.decoder = EnsembleConditionalDynamicsModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            z_dim=self.z_dim,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            ensemble_size=self.ensemble_size,
        ).to(self.device)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.get("encoder_lr", 3e-4),
        )

        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=config.get("decoder_lr", 3e-4),
        )

        self.total_it = 0

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _flatten_trajectory_batch(self, batch, z):
        """
        Convert a padded trajectory batch into transition-level tensors.

        Args:
            batch:
                observations:      [B, T, state_dim]
                actions:           [B, T, action_dim]
                next_observations: [B, T, state_dim]
                mask:              [B, T, 1]
            z:
                trajectory-level latent: [B, z_dim]

        Returns:
            state_flat:        [B*T, state_dim]
            action_flat:       [B*T, action_dim]
            z_flat:            [B*T, z_dim]
            target_delta_flat: [B*T, state_dim]
            mask_flat:         [B*T, 1]
        """
        observations = batch["observations"]
        actions = batch["actions"]
        next_observations = batch["next_observations"]
        mask = batch["mask"]

        batch_size, horizon, state_dim = observations.shape
        z_dim = z.shape[-1]

        delta = next_observations - observations

        z_expand = z[:, None, :].expand(batch_size, horizon, z_dim)

        state_flat = observations.reshape(batch_size * horizon, state_dim)
        action_flat = actions.reshape(batch_size * horizon, actions.shape[-1])
        z_flat = z_expand.reshape(batch_size * horizon, z_dim)
        target_delta_flat = delta.reshape(batch_size * horizon, state_dim)
        mask_flat = mask.reshape(batch_size * horizon, 1)

        return state_flat, action_flat, z_flat, target_delta_flat, mask_flat

    def decoder_loss(self, batch, z):
        """
        Masked ensemble MSE loss for dynamics reconstruction.

        pred_delta:
            [ensemble_size, B*T, state_dim]

        target_delta_flat:
            [B*T, state_dim]

        mask_flat:
            [B*T, 1]
        """
        state_flat, action_flat, z_flat, target_delta_flat, mask_flat = \
            self._flatten_trajectory_batch(batch, z)

        pred_delta = self.decoder(state_flat, action_flat, z_flat)
        # [ensemble_size, B*T, state_dim]

        target = target_delta_flat.unsqueeze(0)
        # [1, B*T, state_dim]

        mask = mask_flat.unsqueeze(0)
        # [1, B*T, 1]

        mse = ((pred_delta - target) ** 2) * mask

        denom = (
            mask.sum().clamp(min=1.0)
            * target_delta_flat.shape[-1]
            * pred_delta.shape[0]
        )

        loss = mse.sum() / denom
        return loss

    def e_step(self, batch):
        """
        E-step:
            fix decoder, update encoder.
        """
        self.encoder.train()
        self.decoder.eval()

        self._set_requires_grad(self.encoder, True)
        self._set_requires_grad(self.decoder, False)

        z, mean, log_std = self.encoder.sample(
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["mask"],
        )

        recon_loss = self.decoder_loss(batch, z)
        kl_loss = gaussian_kl_to_standard_normal(mean, log_std)

        loss = recon_loss + self.beta_kl * kl_loss

        self.encoder_optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                self.grad_clip_norm,
            )

        self.encoder_optimizer.step()

        return {
            "encoder_loss": loss.detach(),
            "encoder_recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

    def m_step(self, batch):
        """
        M-step:
            fix encoder, update decoder.
        """
        self.encoder.eval()
        self.decoder.train()

        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.decoder, True)

        with torch.no_grad():
            mean, _ = self.encoder(
                batch["observations"],
                batch["actions"],
                batch["next_observations"],
                batch["mask"],
            )
            z = mean

        recon_loss = self.decoder_loss(batch, z)

        self.decoder_optimizer.zero_grad()
        recon_loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(),
                self.grad_clip_norm,
            )

        self.decoder_optimizer.step()

        return {
            "decoder_loss": recon_loss.detach(),
        }

    def train(self, replay_buffer, batch_size=32, max_len=None, writer=None):
        """
        One TC-ELBO update.

        Args:
            replay_buffer:
                SourceTrajectoryBuffer, must implement sample_trajectories().
            batch_size:
                number of trajectories.
            max_len:
                optional trajectory cropping length.
            writer:
                optional tensorboard writer.

        Returns:
            info dict with detached tensors.
        """
        self.total_it += 1

        batch = replay_buffer.sample_trajectories(
            batch_size=batch_size,
            max_len=max_len,
        )

        e_info = self.e_step(batch)
        m_info = self.m_step(batch)

        if writer is not None and self.total_it % 100 == 0:
            writer.add_scalar(
                "repr/encoder_loss",
                e_info["encoder_loss"].item(),
                self.total_it,
            )
            writer.add_scalar(
                "repr/encoder_recon_loss",
                e_info["encoder_recon_loss"].item(),
                self.total_it,
            )
            writer.add_scalar(
                "repr/decoder_loss",
                m_info["decoder_loss"].item(),
                self.total_it,
            )
            writer.add_scalar(
                "repr/kl_loss",
                e_info["kl_loss"].item(),
                self.total_it,
            )

        info = {}
        info.update(e_info)
        info.update(m_info)
        return info

    @torch.no_grad()
    def infer_z(self, batch):
        """
        Infer trajectory-level representation z using encoder mean.

        Args:
            batch:
                trajectory batch from SourceTrajectoryBuffer.sample_trajectories()

        Returns:
            z: [B, z_dim]
        """
        self.encoder.eval()

        z = self.encoder.infer_mean(
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["mask"],
        )

        return z

    def save(self, filename):
        """
        Save model in DVDF style.

        If filename is './logs/Repr/model', this will create:
            ./logs/Repr/model_encoder
            ./logs/Repr/model_decoder
            ./logs/Repr/model_encoder_optimizer
            ./logs/Repr/model_decoder_optimizer
        """
        torch.save(self.encoder.state_dict(), filename + "_encoder")
        torch.save(self.decoder.state_dict(), filename + "_decoder")
        torch.save(self.encoder_optimizer.state_dict(), filename + "_encoder_optimizer")
        torch.save(self.decoder_optimizer.state_dict(), filename + "_decoder_optimizer")

    def load(self, filename, load_optimizer=True):
        """
        Load model in DVDF style.

        Args:
            filename:
                prefix used in save().
            load_optimizer:
                whether to load optimizer states.
        """
        self.encoder.load_state_dict(
            torch.load(filename + "_encoder", map_location=self.device)
        )
        self.decoder.load_state_dict(
            torch.load(filename + "_decoder", map_location=self.device)
        )

        if load_optimizer:
            self.encoder_optimizer.load_state_dict(
                torch.load(filename + "_encoder_optimizer", map_location=self.device)
            )
            self.decoder_optimizer.load_state_dict(
                torch.load(filename + "_decoder_optimizer", map_location=self.device)
            )