import os
import tqdm
import torch
import importlib
import wandb
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.vis_utils import visualize_point_clouds_3d, \
    visualize_procedure
from trainers.utils.utils import get_opt, get_prior, \
    ground_truth_reconstruct_multi, set_random_seed


try:
    from evaluation.evaluation_metrics import EMD_CD
    eval_reconstruciton = True
except Exception as e:  # noqa
    # Skip evaluation
    print(f'Error: {e}')
    eval_reconstruciton = False


# def score_matching_loss(score_net, shape_latent, tr_pts, sigma):
#     bs, num_pts = tr_pts.size(0), tr_pts.size(1)
#     sigma = sigma.view(bs, 1, 1)
#     perturbed_points = tr_pts + torch.randn_like(tr_pts) * sigma

#     # For numerical stability, the network predicts the field in a normalized
#     # scale (i.e. the norm of the gradient is not scaled by `sigma`)
#     # As a result, when computing the ground truth for supervision, we are using
#     # its original scale without scaling by `sigma`
#     y_pred = score_net(perturbed_points, shape_latent)  # field (B, #points, 3)
#     y_gtr = - (perturbed_points - tr_pts).view(bs, num_pts, -1)

#     # The loss for each sigma is weighted
#     lambda_sigma = 1. / sigma
#     loss = 0.5 * ((y_gtr - y_pred) ** 2. * lambda_sigma).sum(dim=2).mean()
#     return {
#         "loss": loss,
#         "x": perturbed_points
#     }

# def sample_pairs(x1, x0=None):
#     t = torch.rand((x1.shape[0], 1, 1)).to(x1.device)
#     xt = t * x1 + (1.-t) * x0
#     target = x1 - x0
#     return xt, t * 999, target

def get_train_tuple(z0=None, z1=None, n_timesteps=1_000):
    t = torch.rand((z1.shape[0], 1, 1)).to(z1.device)
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0 
    # t = (t * 999).type(torch.int64)
    return z_t, t, target

def flow_matching_loss(vnet, z, data, noise=None):
    B, D, N = data.shape
    if noise is None:
        noise = torch.randn_like(data)
    noise = noise.to(data.device)
    # xt, t, target = sample_pairs(x1=data, x0=noise)
    xt, t, target = get_train_tuple(z0=noise, z1=data)
    t = t.squeeze()
    eps_recon = vnet(xt, z, t)
    loss = ((target - eps_recon)**2).mean(dim=list(range(1, len(data.shape))))
    return {
        "loss": loss.mean(),
        "x": xt
    }

class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        sn_lib = importlib.import_module(cfg.models.scorenet.type)
        self.vnet = sn_lib.Decoder(cfg, cfg.models.scorenet)
        self.vnet.cuda()
        print("VNet:")
        print(self.vnet)

        encoder_lib = importlib.import_module(cfg.models.encoder.type)
        self.encoder = encoder_lib.Encoder(cfg.models.encoder)
        self.encoder.cuda()
        print("Encoder:")
        print(self.encoder)

        # The optimizer
        if not (hasattr(self.cfg.trainer, "opt_enc") and
                hasattr(self.cfg.trainer, "opt_dec")):
            self.cfg.trainer.opt_enc = self.cfg.trainer.opt
            self.cfg.trainer.opt_dec = self.cfg.trainer.opt

        self.opt_enc, self.scheduler_enc = get_opt(
            self.encoder.parameters(), self.cfg.trainer.opt_enc)
        self.opt_dec, self.scheduler_dec = get_opt(
            self.vnet.parameters(), self.cfg.trainer.opt_dec)

        # Sigmas
        # if hasattr(cfg.trainer, "sigmas"):
        #     self.sigmas = cfg.trainer.sigmas
        # else:
        #     self.sigma_begin = float(cfg.trainer.sigma_begin)
        #     self.sigma_end = float(cfg.trainer.sigma_end)
        #     self.num_classes = int(cfg.trainer.sigma_num)
        #     self.sigmas = np.exp(
        #         np.linspace(np.log(self.sigma_begin),
        #                     np.log(self.sigma_end),
        #                     self.num_classes))
        # print("Sigma:, ", self.sigmas)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # Prepare variable for summy
        self.oracle_res = None

    def multi_gpu_wrapper(self, wrapper):
        self.encoder = wrapper(self.encoder)
        self.vnet = wrapper(self.vnet)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step()
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dec_lr', self.scheduler_dec.get_last_lr()[0], epoch)
        if self.scheduler_enc is not None:
            self.scheduler_enc.step()
            if writer is not None:
                writer.add_scalar(
                    'train/opt_enc_lr', self.scheduler_enc.get_last_lr()[0], epoch)

    def update(self, data, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.encoder.train()
            self.vnet.train()
            self.opt_enc.zero_grad()
            self.opt_dec.zero_grad()

        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        batch_size = tr_pts.size(0)
        z_mu, _ = self.encoder(tr_pts)
        z = z_mu

        # Randomly sample sigma
        # labels = torch.randint(
        #     0, len(self.sigmas), (batch_size,), device=tr_pts.device)
        # used_sigmas = torch.tensor(
        #     np.array(self.sigmas))[labels].float().view(batch_size, 1).cuda()
        # z = torch.cat((z, used_sigmas), dim=1)

        noise = torch.randn(batch_size, tr_pts.shape[1], tr_pts.shape[2])
        noise = noise.to(tr_pts.device)
        res = flow_matching_loss(self.vnet, z, tr_pts, noise)
        loss = res['loss']
        if not no_update:
            loss.backward()
            self.opt_enc.step()
            self.opt_dec.step()

        return {
            'loss': loss.detach().cpu().item(),
            'x': res['x'].detach().cpu()            # perturbed data
        }

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            print("No writer, exiting...")
            return

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            if step is not None:
                writer.add_scalar('train/' + k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar('train/' + k, v, epoch)

        if visualize:
            with torch.no_grad():
                print("Visualize: %s" % step)
                gtr = train_data['te_points']  # ground truth point cloud
                inp = train_data['tr_points']  # input for encoder
                # ptb = train_info['x']  # perturbed data
                num_vis = min(
                    getattr(self.cfg.viz, "num_vis_samples", 5),
                    gtr.size(0))

                print("Recon:")
                rec, rec_list, timestamps = self.reconstruct(inp=inp[:num_vis].cuda(), n_timesteps=1000)
                # print("Ground truth recon:")
                # rec_gt, rec_gt_list = ground_truth_reconstruct_multi(
                #     inp[:num_vis].cuda(), self.cfg)

                print(f'Gtr: min: {gtr[:num_vis].min()} max: {gtr[:num_vis].max()}, mean: {gtr[:num_vis].mean()}')
                print(f'Rec: min: {rec.min()} max: {rec.max()}, mean: {rec.mean()}')

                print("Saving Reconstructed point clouds")
                generated = [rec[idx].cpu().detach().numpy() for idx in range(num_vis)]
                ground_truth = [gtr[idx].cpu().detach().numpy() for idx in range(num_vis)]
                # ground_truth = [rec_gt[idx].cpu().detach().numpy() for idx in range(num_vis)]
                
                wandb.log({ "Reconstructed": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in generated],
                            "True Shape": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in ground_truth]})

                # Overview
                all_imgs = []
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d(
                        # [rec_gt[idx], rec[idx], gtr[idx], ptb[idx]],
                        [rec[idx], gtr[idx]],
                        ["Reconstruction", "True Shape"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image(
                    'tr_vis/overview', torch.as_tensor(img), step)

                # # Reconstruction gt procedure
                # img = visualize_procedure(
                #     self.sigmas, rec_gt_list, gtr, num_vis, self.cfg, "Rec_gt")
                # writer.add_image(
                #     'tr_vis/rec_gt_process', torch.as_tensor(img), step)

                # Reconstruction procedure
                img = visualize_procedure(
                    timestamps, rec_list, gtr, num_vis, self.cfg, "Rec")
                writer.add_image(
                    'tr_vis/rec_process', torch.as_tensor(img), step)

    def validate(self, test_loader, epoch, *args, **kwargs):
        if not eval_reconstruciton:
            return {}

        print("Validation (reconstruction):")
        all_ref, all_rec, all_smp, all_ref_denorm = [], [], [], []
        all_rec_gt, all_inp_denorm, all_inp = [], [], []
        for data in tqdm.tqdm(test_loader):
            ref_pts = data['te_points'].cuda()
            inp_pts = data['tr_points'].cuda()
            m = data['mean'].cuda()
            std = data['std'].cuda()
            print(f'm: {m}, std: {std}')
            rec_pts, _, _ = self.reconstruct(inp_pts, save_img_freq=1000)

            # denormalize
            inp_pts_denorm = inp_pts.clone() * std + m
            ref_pts_denorm = ref_pts.clone() * std + m
            rec_pts = rec_pts * std + m

            all_inp.append(inp_pts)
            all_inp_denorm.append(inp_pts_denorm.view(*inp_pts.size()))
            all_ref_denorm.append(ref_pts_denorm.view(*ref_pts.size()))
            all_rec.append(rec_pts.view(*ref_pts.size()))
            all_ref.append(ref_pts)

        inp = torch.cat(all_inp, dim=0)
        rec = torch.cat(all_rec, dim=0)
        ref = torch.cat(all_ref, dim=0)
        ref_denorm = torch.cat(all_ref_denorm, dim=0)
        inp_denorm = torch.cat(all_inp_denorm, dim=0)
        for name, arr in [
            ('inp', inp), ('rec', rec), ('ref', ref),
            ('ref_denorm', ref_denorm), ('inp_denorm', inp_denorm)]:
            np.save(
                os.path.join(
                    self.cfg.save_dir, 'val', '%s_ep%d.npy' % (name, epoch)),
                arr.detach().cpu().numpy()
            )
        all_res = {}

        # Oracle CD/EMD, will compute only once
        if self.oracle_res is None:
            rec_res = EMD_CD(inp_denorm, ref_denorm, 1)
            rec_res = {
                ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
                for k, v in rec_res.items()}
            all_res.update(rec_res)
            print("Validation oracle (denormalize) Epoch:%d " % epoch, rec_res)
            self.oracle_res = rec_res
        else:
            all_res.update(self.oracle_res)

        # Reconstruction CD/EMD
        all_res = {}
        rec_res = EMD_CD(rec, ref_denorm, 1)
        rec_res = {
            ("val/rec/%s" % k): (v if isinstance(v, float) else v.item())
            for k, v in rec_res.items()}
        all_res.update(rec_res)
        print("Validation Recon (denormalize) Epoch:%d " % epoch, rec_res)

        return all_res

    def save(self, epoch=None, step=None, appendix=None, wandb_run=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            'vn': self.vnet.state_dict(),
            'enc': self.encoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)
        if wandb_run is not None:
            artifact = wandb.Artifact(save_name, type='model')
            artifact.add_file(path)
            wandb_run.log_artifact(artifact)

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.vnet.load_state_dict(ckpt['vn'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    # def langevin_dynamics(self, z, num_points=2048):
    #     with torch.no_grad():
    #         assert hasattr(self.cfg, "inference")
    #         step_size_ratio = float(getattr(
    #             self.cfg.inference, "step_size_ratio", 1))
    #         num_steps = int(getattr(self.cfg.inference, "num_steps", 5))
    #         num_points = int(getattr(
    #             self.cfg.inference, "num_points", num_points))
    #         weight = float(getattr(self.cfg.inference, "weight", 1))
    #         sigmas = self.sigmas

    #         x_list = []
    #         self.score_net.eval()
    #         x = get_prior(z.size(0), num_points, self.cfg.models.scorenet.dim)
    #         x = x.to(z)
    #         x_list.append(x.clone())
    #         for sigma in sigmas:
    #             sigma = torch.ones((1,)).cuda() * sigma
    #             z_sigma = torch.cat((z, sigma.expand(z.size(0), 1)), dim=1)
    #             step_size = 2 * sigma ** 2 * step_size_ratio
    #             for t in range(num_steps):
    #                 z_t = torch.randn_like(x) * weight
    #                 x += torch.sqrt(step_size) * z_t
    #                 grad = self.score_net(x, z_sigma)
    #                 grad = grad / sigma ** 2
    #                 x += 0.5 * step_size * grad
    #             x_list.append(x.clone())
    #     return x, x_list
    def new_x_chain(self, x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    def generate_sample(self, z, num_points, n_timesteps, save_img_freq=250):
        print(f'z shape: {z.shape}')
        img_t = get_prior(z.size(0), num_points, self.cfg.models.scorenet.dim)
        img_t = img_t.to(z)
        imgs = []
        timestamps = []
        with torch.no_grad():
            self.vnet.eval()
            for t in range(n_timesteps):
                t_ = torch.empty(z.shape[0], dtype=torch.int64, device=z.device).fill_(t)
                img_t = img_t + self.vnet(img_t, z, t_) * 1. / n_timesteps
                if (t + 1) % save_img_freq == 0:
                    imgs.append(img_t.clone())
                    timestamps.append(t)
        return img_t, imgs, timestamps

    def sample(self, num_shapes=1, num_points=2048):
        with torch.no_grad():
            # z = torch.randn(num_shapes, self.cfg.models.encoder.zdim).cuda()
            return self.generate_sample(z, num_points=num_points)

    def reconstruct(self, inp, num_points=2048, n_timesteps=1000, save_img_freq=200):
        with torch.no_grad():
            self.encoder.eval()
            z, _ = self.encoder(inp)
            return self.generate_sample(z, num_points=num_points, n_timesteps=n_timesteps, save_img_freq=save_img_freq)

