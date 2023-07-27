import torch
import torch.nn as nn
import numpy as np

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, t_dim: int):
        super().__init__()
        self.t_dim = t_dim
        self.lin1 = nn.Linear(self.t_dim // 4, self.t_dim)
        self.act = Swish()
        self.lin2 = nn.Linear(self.t_dim, self.t_dim)

    def forward(self, t: torch.Tensor):
        half_dim = self.t_dim // 8
        emb = np.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False, t_dim=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if norm_method == 'batch_norm':
            norm = nn.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = nn.SyncBatchNorm
        else:
             raise Exception("Invalid norm method: %s" % norm_method)

        self.bn_0 = norm(size_in)
        self.bn_1 = norm(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.fc_c = nn.Conv1d(c_dim + t_dim, size_out, 1)
        self.actvn = nn.ReLU()

        self.time_emb = nn.Linear(t_dim, size_h)
        self.time_act = Swish()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c, t):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        print(f't: {t.shape}')
        time = self.time_emb(self.time_act(t))
        # print(f'net in forward: {net.shape}')
        # print(f'time in forward: {time[:, :, None].shape}')
        net += time[:, :, None]
        # dx = self.fc_1(self.actvn(self.bn_1(net)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(torch.cat([c, time], dim=1))

        return out


class Decoder(nn.Module):
    """ Decoder conditioned by adding.

    Example configuration:
        z_dim: 128
        hidden_size: 256
        n_blocks: 5
        out_dim: 3  # we are outputting the gradient
        sigma_condition: True
        xyz_condition: True
    """

    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.z_dim = z_dim = cfg.z_dim
        self.t_dim = t_dim = cfg.t_dim
        self.dim = dim = cfg.dim
        self.out_dim = out_dim = cfg.out_dim
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks

        # Input = Conditional = zdim (shape) + dim (xyz) + tdim (time)
        self.time_emb = TimeEmbedding(t_dim * 4)
        c_dim = z_dim + dim
        self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockConv1d(c_dim, hidden_size, t_dim=t_dim) for _ in range(n_blocks)
        ])
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
        self.actvn_out = nn.ReLU()

    # This should have the same signature as the sig condition one
    def forward(self, x, c, t):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim + 1) Shape latent code + sigma
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        p = x.transpose(1, 2)  # (bs, dim, n_points)
        batch_size, D, num_points = p.size()
        t = self.time_emb(t)
        # time_emb = self.get_timestep_embedding(t, t.device)  # (B, 1, tdim)
        # time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # (B, 1, 3)
        #print(f'c: {c.shape}, time_emb: {time_emb.shape}')
        # ctx_emb = torch.cat([c, time_emb], dim=-1) # p: torch.Size([32, 3, 2048]), ctx: torch.Size([32, 1, 131])

        c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
        # time_emb = time_emb.unsqueeze(2).expand(-1, -1, num_points)
        #print(f'p: {p.shape}, ctx_emb: {ctx_emb.shape}, c_expand: {c_expand.shape}')
        c_xyz = torch.cat([p, c_expand], dim=1)
        net = self.conv_p(c_xyz)
        for block in self.blocks:
            net = block(net, c_xyz, t)
        out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
        return out
    # def __init__(self, _, cfg):
    #     super().__init__()
    #     self.cfg = cfg
    #     self.z_dim = z_dim = cfg.z_dim
    #     self.t_dim = t_dim = cfg.t_dim
    #     self.dim = dim = cfg.dim
    #     self.out_dim = out_dim = cfg.out_dim
    #     self.hidden_size = hidden_size = cfg.hidden_size
    #     self.n_blocks = n_blocks = cfg.n_blocks

    #     # Input = Conditional = zdim (shape) + dim (xyz)
    #     c_dim = self.z_dim + dim
    #     r_dim = c_dim + self.t_dim
    #     #print(f'cdim: {c_dim}')
    #     self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
    #     self.blocks = nn.ModuleList([
    #         ResnetBlockConv1d(r_dim, hidden_size) for _ in range(n_blocks)
    #     ])
    #     self.bn_out = nn.BatchNorm1d(hidden_size)
    #     self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
    #     self.actvn_out = nn.ReLU()

    # # This should have the same signature as the sig condition one
    # def forward(self, x, c, t):
    #     """
    #     :param x: (bs, npoints, self.dim) Input coordinate (xyz)
    #     :param c: (bs, self.zdim) Shape latent code
    #     :param t: (bs, self.tdim) Time embedding
    #     :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
    #     """
    #     print(f't: {t.shape}')
    #     p = x.transpose(1, 2)  # (bs, dim, n_points)
    #     batch_size, D, num_points = p.size()

    #     c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
    #     time_emb = self.get_timestep_embedding(t, t.device)
    #     c_xyz = torch.cat([p, c_expand], dim=1)
    #     #print(f'p: {p.shape} c_xyz: {c_xyz.shape} dim: {self.dim} zdim: {self.z_dim}')
    #     net = self.conv_p(c_xyz)

    #     # t_expand = time_emb.unsqueeze(2).expand(-1, -1, num_points)
    #     r_xyz = torch.cat([p, c_expand, time_emb], dim=1)
    #     #print(f'temb: {t_expand.shape} r_xyz: {r_xyz.shape} dim: {self.dim} zdim: {self.t_dim}')

    #     for block in self.blocks:
    #         net = block(net, r_xyz)
    #     out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
    #     return out

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.t_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.t_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.t_dim])
        return emb