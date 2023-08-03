import os
import tqdm
import torch
import random
import importlib
import wandb
import numpy as np
from trainers.utils.utils import get_opt
from trainers.ae_trainer_3D import Trainer as BaseTrainer
from trainers.utils.gan_losses import gen_loss, dis_loss, gradient_penalty
from trainers.utils.vis_utils import visualize_procedure, \
    visualize_point_clouds_3d


try:
    from evaluation.evaluation_metrics import compute_all_metrics

    eval_generation = True
except:  # noqa
    eval_generation = False

def get_train_tuple(z0=None, z1=None, n_timesteps=1_000):
    t = torch.rand((z1.shape[0], 1, 1)).to(z1.device)
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0 
    return z_t, t * 999, target

def flow_matching_loss(vnet, data, noise=None):
    # B, D, N = data.shape
    if noise is None:
        noise = torch.randn_like(data)
    noise = noise.to(data.device)
    xt, t, target = get_train_tuple(z0=noise, z1=data)
    t = t.squeeze()
    model_output = vnet(xt, t)
    sqerr = (target - model_output)**2
    loss = sqerr.mean()
    return {
        "loss": loss,
        "x": xt
    }

class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        # Now initialize the GAN part
        gen_lib = importlib.import_module(cfg.models.gen.type)
        self.gen = gen_lib.Generator(cfg, cfg.models.gen)
        self.gen.cuda()
        print("Generator:")
        print(self.gen)

        # dis_lib = importlib.import_module(cfg.models.dis.type)
        # self.dis = dis_lib.Discriminator(cfg, cfg.models.dis)
        # self.dis.cuda()
        # print("Discriminator:")
        # print(self.dis)

        # Optimizers
        if not (hasattr(self.cfg.trainer, "opt_gen") and
                hasattr(self.cfg.trainer, "opt_dis")):
            self.cfg.trainer.opt_gen = self.cfg.trainer.opt
            # self.cfg.trainer.opt_dis = self.cfg.trainer.opt
        self.opt_gen, self.scheduler_gen = get_opt(
            self.gen.parameters(), self.cfg.trainer.opt_gen)
        # self.opt_dis, self.scheduler_dis = get_opt(
        #     self.dis.parameters(), self.cfg.trainer.opt_dis)

        # book keeping
        # self.total_iters = 0
        # self.total_gan_iters = 0
        self.gan_only = getattr(self.cfg.trainer, "gan_only", True)

        # If pretrained AE, then load it up
        if hasattr(self.cfg.trainer, "ae_pretrained"):
            ckpt = torch.load(self.cfg.trainer.ae_pretrained)
            print(self.cfg.trainer.ae_pretrained)
            strict = getattr(self.cfg.trainer, "resume_strict", True)
            self.encoder.load_state_dict(ckpt['enc'], strict=strict)
            self.vnet.load_state_dict(ckpt['vn'], strict=strict)
            if getattr(self.cfg.trainer, "resume_opt", False):
                self.opt_enc.load_state_dict(ckpt['opt_enc'])
                self.opt_dec.load_state_dict(ckpt['opt_dec'])
        self.gan_pass_update_enc = getattr(
            self.cfg.trainer, "gan_pass_update_enc", False)

    def epoch_end(self, epoch, writer=None, **kwargs):
        super().epoch_end(epoch, writer=writer)

        if self.scheduler_dis is not None:
            self.scheduler_dis.step()
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dis_lr', self.scheduler_dis.get_last_lr()[0], epoch)

        if self.scheduler_gen is not None:
            self.scheduler_gen.step()
            if writer is not None:
                writer.add_scalar(
                    'train/opt_gen_lr', self.scheduler_gen.get_last_lr()[0], epoch)

    def _update(self, data, gen=False):
        self.gen.train()
        self.opt_gen.zero_grad()
        if self.gan_pass_update_enc:
            self.encoder.train()
            self.opt_enc.zero_grad()
        else:
            self.encoder.eval()

        tr_pts = data['tr_points'].cuda()  # (B, #points, 3)smn_ae_trainer.py
        c, _ = self.encoder(tr_pts)
        batch_size = c.size(0)

        noise = torch.randn(*c.shape)
        noise = noise.to(c.device)
        res = flow_matching_loss(self.gen, c, noise)
        loss = res['loss']
        if not no_update:
            loss.backward()
            self.opt_enc.step()
            self.opt_dec.step()

        return {
            'loss': loss.detach().cpu().item(),
            'x': res['x'].detach().cpu()
        }

    # def update_lgan(self, data):
    #     self.total_gan_iters += 1
    #     res = {}
    #     if self.total_gan_iters % self.n_critics == 0:
    #         gen_res = self._update_gan_(data, gen=True)
    #         res.update(gen_res)
    #     dis_res = self._update_gan_(data, gen=False)
    #     res.update(dis_res)
    #     return res

    def update(self, data, *args, **kwargs):
        res = {}
        if not self.gan_only:
            ae_res = super().update(data, *args, **kwargs)
            res.update(ae_res)
        # gan_res = self.update_lgan(data)
        gan_res = self._update(data)
        res.update(gan_res)
        return res

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        with torch.no_grad():
            train_info.update(super().update(train_data, no_update=True))
        super().log_train(train_info, train_data, writer=writer, step=step,
                          epoch=epoch, visualize=visualize)

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
                print("Visualize generation results: %s" % step)
                gtr = train_data['te_points']  # ground truth point cloud
                inp = train_data['tr_points']  # input for encoder
                num_vis = min(
                    getattr(self.cfg.viz, "num_vis_samples", 5),
                    gtr.size(0)
                )
                smp, smp_list, timestamps = self.sample(num_shapes=num_vis,
                                            num_points=inp.size(1))

                all_imgs = []
                for idx in range(num_vis):
                    img = visualize_point_clouds_3d(
                        [smp[idx], gtr[idx]], ["gen", "ref"])
                    all_imgs.append(img)
                img = np.concatenate(all_imgs, axis=1)
                writer.add_image(
                    'tr_vis/gen', torch.as_tensor(img), step)

                img = visualize_procedure(
                     timestamps, smp_list, gtr, num_vis, self.cfg, "gen")
                    # self.sigmas, smp_list, gtr, num_vis, self.cfg, "gen")
                writer.add_image(
                    'tr_vis/gen_process', torch.as_tensor(img), step)
                print("Saving point clouds")
                generated = [smp[idx].cpu().detach().numpy() for idx in range(num_vis)]
                ground_truth = [gtr[idx].cpu().detach().numpy() for idx in range(num_vis)]
                wandb.log({
                    "generated_samples": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in generated],
                    "gtr": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in ground_truth]
                    })

    def save(self, epoch=None, step=None, appendix=None, wandb_run=None, **kwargs):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dec': self.opt_dec.state_dict(),
            # 'opt_dis': self.opt_dis.state_dict(),
            'opt_gen': self.opt_gen.state_dict(),
            'vn': self.vnet.state_dict(),
            'enc': self.encoder.state_dict(),
            # 'dis': self.dis.state_dict(),
            'gen': self.gen.state_dict(),
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

    def resume(self, path, strict=True, **args):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.vnet.load_state_dict(ckpt['vn'], strict=strict)
        self.opt_enc.load_state_dict(ckpt['opt_enc'])
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']

        if 'gen' in ckpt:
            self.gen.load_state_dict(ckpt['gen'], strict=strict)
        # if 'dis' in ckpt:
        #     self.dis.load_state_dict(ckpt['dis'], strict=strict)
        if 'opt_gen' in ckpt:
            self.opt_gen.load_state_dict(ckpt['opt_gen'])
        # if 'opt_dis' in ckpt:
        #     self.opt_dis.load_state_dict(ckpt['opt_dis'])
        return start_epoch

    def sample(self, num_shapes=1, num_points=2048):
        with torch.no_grad():
            self.gen.eval()
            z = self.gen(bs=num_shapes)
            noise = torch.randn((z.shape[0], num_points, 3))
            return self.generate_sample(z, noise=noise, num_points=num_points)

    def validate(self, test_loader, epoch, *args, **kwargs):
        all_res = {}
        done_interp = False
        eval_generation = True
        if eval_generation:
            with torch.no_grad():
                all_ref, all_smp = [], []
                for data in tqdm.tqdm(test_loader):
                    print("l-flow validation:")
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    smp_pts, _, _ = self.sample(
                        num_shapes=inp_pts.size(0),
                        num_points=inp_pts.size(1),
                    )
                    all_smp.append(smp_pts.view(
                        ref_pts.size(0), ref_pts.size(1), ref_pts.size(2)))
                    all_ref.append(
                        ref_pts.view(ref_pts.size(0), ref_pts.size(1),
                                     ref_pts.size(2)))

                smp = torch.cat(all_smp, dim=0)
                np.save(
                    os.path.join(self.cfg.save_dir, 'val',
                                 'smp_ep%d.npy' % epoch),
                    smp.detach().cpu().numpy()
                )
                ref = torch.cat(all_ref, dim=0)

                # Sample CD/EMD
                # step 1: subsample shapes
                max_gen_vali_shape = int(getattr(
                    self.cfg.trainer, "max_gen_validate_shapes",
                    int(smp.size(0))))
                sub_sampled = random.sample(
                    range(smp.size(0)), min(smp.size(0), max_gen_vali_shape))
                smp_sub = smp[sub_sampled, ...].contiguous()
                ref_sub = ref[sub_sampled, ...].contiguous()

                gen_res = compute_all_metrics(
                    smp_sub, ref_sub,
                    batch_size=int(getattr(
                        self.cfg.trainer, "val_metrics_batch_size", 100)),
                    accelerated_cd=True
                )
                all_res = {
                    ("val/gen/%s" % k):
                        (v if isinstance(v, float) else v.item())
                    for k, v in gen_res.items()}
                print("Validation Sample (unit) Epoch:%d " % epoch, gen_res)


        # Call super class validation
        if getattr(self.cfg.trainer, "validate_recon", False):
            all_res.update(super().validate(
                test_loader, epoch, *args, **kwargs))

        return all_res

    def interpolate(self, bs, *args, **kwargs):
        all_res = {}
        with torch.no_grad():
            print("l-flow interpolation:")
            self.gen.eval()
            z1 = self.gen(bs=bs)
            z2 = self.gen(bs=bs)
            taus = torch.arange(0, 21) * 0.05
            for tau in taus:
                z = (z1 * (1-tau) + z2*tau)/((1-tau)**2 + tau**2)

                print(f'z shape: {z.shape}')
                samples, _, _ = self.generate_sample(z=z)
                print(f'z shape: {z.shape}')
                generated = [samples[idx].cpu().detach().numpy() for idx in range(z.shape[0])]
                # print(f'prior: {prior}')
                # zs = [z[idx].cpu().detach().numpy() for idx in range(z.shape[0])]
                wandb.log({
                    "generated_samples": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in generated]
                    })
        return all_res