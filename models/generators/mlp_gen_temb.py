import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class Generator(nn.Module):

    def __init__(self, cfg, cfgmodel):
        super().__init__()
        self.cfg = cfg
        self.cfgmodel = cfgmodel
        self.inp_dim = cfgmodel.inp_dim
        self.out_dim = cfgmodel.out_dim
        self.use_bn = getattr(cfgmodel, "use_bn", False)
        self.output_bn = getattr(cfgmodel, "output_bn", False)
        self.dims = cfgmodel.dims
        self.t_dim = cfgmodel.t_dim
        curr_dim = self.inp_dim + self.t_dim
        self.layers = []
        self.bns = []
        for hid in self.dims:
            self.layers.append(nn.Linear(curr_dim, hid))
            self.bns.append(nn.BatchNorm1d(hid))
            curr_dim = hid
        self.layers = nn.ModuleList(self.layers)
        self.bns = nn.ModuleList(self.bns)
        self.out = nn.Linear(curr_dim, self.out_dim)
        self.out_bn = nn.BatchNorm1d(self.out_dim)
        self.prior_type = getattr(cfgmodel, "prior", "gaussian")

    # def get_prior(self, bs):
    #     if self.prior_type == "truncate_gaussian":
    #         gaussian_scale = getattr(self.cfgmodel, "gaussian_scale", 1.)
    #         truncate_std = getattr(self.cfgmodel, "truncate_std", 2.)
    #         noise = (torch.randn(bs, self.inp_dim) * gaussian_scale).cuda()
    #         noise = truncated_normal(
    #             noise, mean=0, std=gaussian_scale, trunc_std=truncate_std)
    #         return noise
    #     elif self.prior_type == "gaussian":
    #         gaussian_scale = getattr(self.cfgmodel, "gaussian_scale", 1.)
    #         return torch.randn(bs, self.inp_dim).cuda() * gaussian_scale

    #     else:
    #         raise NotImplementedError(
    #             "Invalid prior type:%s" % self.prior_type)

    def forward(self, c, t, bs=None):
        # if z is None:
        #     assert bs is not None
        #     z = self.get_prior(bs).cuda()
        time_emb = self.get_timestep_embedding(t, t.device)
        time_emb = time_emb.unsqueeze()
        print(f'c: {c.shape} time_emb: {time_emb.shape}')
        z = torch.cat([c, time_emb], dim=-1)
        y = z
        for layer, bn in zip(self.layers, self.bns):
            y = layer(y)
            if self.use_bn:
                y = bn(y)
            y = F.relu(y)
        y = self.out(y)

        if self.output_bn:
            y = self.out_bn(y)
        return y

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

