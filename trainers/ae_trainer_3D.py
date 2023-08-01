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
    from metrics.evaluation_metrics import EMD_CD
    eval_reconstruciton = True
    print(f'Imported EMD_CD: {eval_reconstruciton}')
except Exception as e:  # noqa
    # Skip evaluation
    print(f'******** EVAL WARNING ******* \n{e}')
    print(f'EXITING')
    eval_reconstruciton = False
    exit(0)

def get_train_tuple(z0=None, z1=None, n_timesteps=1_000):
    t = torch.rand((z1.shape[0], 1, 1)).to(z1.device)
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0 
    return z_t, t * 999, target

def flow_matching_loss(vnet, z, data, noise=None):
    B, D, N = data.shape
    if noise is None:
        noise = torch.randn_like(data)
    noise = noise.to(data.device)
    xt, t, target = get_train_tuple(z0=noise, z1=data)
    t = t.squeeze()
    model_output = vnet(xt, z, t)
    sqerr = (target - model_output)**2
    loss = sqerr.sum(dim=2).mean()
    return {
        "loss": loss,
        "x": xt
    }

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 42))

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
            'x': res['x'].detach().cpu()
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

                print("Saving Reconstructed point clouds")
                generated = [rec[idx].cpu().detach().numpy() for idx in range(num_vis)]
                ground_truth = [gtr[idx].cpu().detach().numpy() for idx in range(num_vis)]
                
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
            print(f'No metrics evaluator. Exiting validation...')
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

    def test_recon(self, test_loader, epoch, *args, **kwargs):
        num_vis = 5
        for data in tqdm.tqdm(test_loader):
            ref_pts = data['te_points'].cuda()
            inp_pts = data['tr_points'].cuda()
            # rec_pts, _, _ = self.reconstruct(inp=inp[:num_vis].cuda(), n_timesteps=1000, save_img_freq=1000)

            rec, rec_list, timestamps = self.reconstruct(inp=inp_pts[:num_vis].cuda(), n_timesteps=1000, save_img_freq=1000)
            # print("Ground truth recon:")
            # rec_gt, rec_gt_list = ground_truth_reconstruct_multi(
            #     inp[:num_vis].cuda(), self.cfg)

            print("Saving Reconstructed point clouds")
            generated = [rec[idx].cpu().detach().numpy() for idx in range(num_vis)]
            ground_truth = [ref_pts[idx].cpu().detach().numpy() for idx in range(num_vis)]
            
            wandb.log({ "Reconstructed": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in generated],
                        "True Shape": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in ground_truth]})


   
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

    def generate_sample(self, z, noise=None, num_points=2048, n_timesteps=1000, save_img_freq=None):
        if noise is None:
            noise = torch.randn((z.size(0), num_points, self.cfg.models.scorenet.dim), dtype=torch.float, device=z.device)
        img_t = noise.clone()
        img_t = img_t.to(z.device)
        imgs = []
        timestamps = []
        dt = 1. / n_timesteps
        with torch.no_grad():
            self.vnet.eval()
            for t in range(n_timesteps):
                t_ = torch.empty(z.shape[0], dtype=torch.int64, device=z.device).fill_(t)
                img_t = img_t + self.vnet(img_t, z, t_) * dt
                if save_img_freq is not None and (t + 1) % save_img_freq == 0:
                    imgs.append(img_t.clone())
                    timestamps.append(t)
        return img_t, imgs, timestamps

    def sample(self, num_shapes=1, num_points=2048):
        with torch.no_grad():
            # z = torch.randn(num_shapes, self.cfg.models.encoder.zdim).cuda()
            noise = torch.randn((z.size(0), num_points, self.cfg.models.scorenet.dim), dtype=torch.float, device=z.device)
            return self.generate_sample(z, noise=noise, num_points=num_points)
    
    def gen_reflow_pairs(self, data, *args, **kwargs):
        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        batch_size = tr_pts.size(0)
        num_points = self.cfg.inference.num_points
        dim = self.cfg.models.scorenet.dim
        with torch.no_grad():
            self.encoder.eval()
            z, _ = self.encoder(tr_pts)
            x0 = torch.randn((z.size(0), num_points, dim), dtype=torch.float, device=z.device)
            x1, _, _ = self.generate_sample(z, noise=x0, num_points=num_points, n_timesteps=1000)
            return [x0, x1, z]

    def reconstruct(self, inp, num_points=2048, n_timesteps=1000, save_img_freq=200):
        with torch.no_grad():
            self.encoder.eval()
            z, _ = self.encoder(inp)
            noise = torch.randn((z.size(0), num_points, self.cfg.models.scorenet.dim), dtype=torch.float, device=z.device)
            return self.generate_sample(z, noise=noise, num_points=num_points, n_timesteps=n_timesteps, save_img_freq=save_img_freq)