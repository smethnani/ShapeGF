import torch
from torch.nn import Module, Linear, ModuleList
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# class ConcatSquashLinear(Module):
#     def __init__(self, dim_in, dim_out, dim_ctx):
#         super(ConcatSquashLinear, self).__init__()
#         self._layer = Linear(dim_in, dim_out)
#         self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
#         self._hyper_gate = Linear(dim_ctx, dim_out)

#     def forward(self, ctx, x):
#         gate = torch.sigmoid(self._hyper_gate(ctx))
#         bias = self._hyper_bias(ctx)
#         ret = self._layer(x) * gate + bias
#         return ret

# class Decoder(Module):

#     def __init__(self, _, cfg): # point_dim, context_dim, residual):
#         super().__init__()
#         self.cfg = cfg
#         point_dim = 3
#         context_dim = cfg.z_dim
#         residual = True
#         self.act = F.leaky_relu
#         self.residual = residual
#         self.layers = ModuleList([
#             ConcatSquashLinear(3, 128, context_dim+3),
#             ConcatSquashLinear(128, 256, context_dim+3),
#             ConcatSquashLinear(256, 512, context_dim+3),
#             ConcatSquashLinear(512, 256, context_dim+3),
#             ConcatSquashLinear(256, 128, context_dim+3),
#             ConcatSquashLinear(128, 3, context_dim+3)
#         ])

#     def forward(self, x, context, beta):
#         """
#         Args:
#             x:  Point clouds at some timestep t, (B, N, d).
#             beta:     Time. (B, ).
#             context:  Shape latents. (B, F).
#         """
#         batch_size = x.size(0)
#         beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
#         context = context.view(batch_size, 1, -1)   # (B, 1, F)

#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
#         ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

#         out = x
#         for i, layer in enumerate(self.layers):
#             out = layer(ctx=ctx_emb, x=out)
#             if i < len(self.layers) - 1:
#                 out = self.act(out)

#         if self.residual:
#             return x + out
#         else:
#             return out

class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
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
        self.fc_c = nn.Conv1d(c_dim, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(c)

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
        c_dim = z_dim + dim + t_dim
        self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockConv1d(c_dim, hidden_size) for _ in range(n_blocks)
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

        time_emb = self.get_timestep_embedding(t, t.device)  # (B, 1, tdim)
        # time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # (B, 1, 3)
        #print(f'c: {c.shape}, time_emb: {time_emb.shape}')
        ctx_emb = torch.cat([c, time_emb], dim=-1) # p: torch.Size([32, 3, 2048]), ctx: torch.Size([32, 1, 131])

        c_expand = ctx_emb.unsqueeze(2).expand(-1, -1, num_points)
        #print(f'p: {p.shape}, ctx_emb: {ctx_emb.shape}, c_expand: {c_expand.shape}')
        c_xyz = torch.cat([p, c_expand], dim=1)
        net = self.conv_p(c_xyz)
        for block in self.blocks:
            net = block(net, c_xyz)
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