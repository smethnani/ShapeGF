import os
import yaml
import time
import torch
import argparse
import importlib
import wandb
import torch.distributed
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from shutil import copy2

def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Flow-based Point Cloud Generation Experiment')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    parser.add_argument('--log', type=str,
                        help='Log destination')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    logdir = args.log
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    if logdir:
        config.log_name = f"{logdir}/gen-reflow-samples-{run_time}"
        config.save_dir = f"{logdir}/gen-reflow-samples-{run_time}"
        config.log_dir = f"{logdir}/gen-reflow-samples-{run_time}"
    else:
        # Currently save dir and log_dir are the same
        config.log_name = "logs/%s_%s" % (cfg_file_name, run_time)
        config.save_dir = "logs/%s_%s" % (cfg_file_name, run_time)
        config.log_dir = "logs/%s_%s" % (cfg_file_name, run_time)

    os.makedirs(config.log_dir+'/config')
    copy2(args.config, config.log_dir+'/config')
    return args, config


def main_worker(cfg, args, wandb_run=None):
    # basic setup
    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=cfg.log_name)
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    start_epoch = 0
    start_time = time.time()

    trainer.resume(args.pretrained)
    # main sampling loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    step = 0
    for epoch in range(1):
        # train for one epoch
        for bidx, data in enumerate(train_loader):
            step = bidx + len(train_loader) * epoch + 1
            # logs_info = trainer.update(data)
            sample_pair = trainer.gen_reflow_pairs(data)
            duration = time.time() - start_time
            print("Epoch %d Batch [%2d/%2d] Time [%3.2fs]"
                    % (epoch, bidx, len(train_loader), duration))
            torch.save(sample_pair[0], f"{cfg.log_dir}/noise-{bidx}.pt")
            torch.save(sample_pair[1], f"{cfg.log_dir}/sample-{bidx}.pt")
            if step % int(cfg.viz.viz_freq) == 0 or epoch < 5:
                num_vis = 5
                x0, x1 = sample_pair
                noise = [x0[idx].cpu().detach().numpy() for idx in range(num_vis)]
                samples = [x1[idx].cpu().detach().numpy() for idx in range(num_vis)]
                data_viz = data['tr_points'][:num_vis]
                true_samples = [data_viz[idx].cpu().detach().numpy() for idx in range(num_vis)]

                wandb.log({ "Sample": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in samples],
                            "True": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in true_samples],
                            "Noise": [wandb.Object3D(pc[:, [0, 2, 1]]) for pc in noise]})
            start_time = time.time()

    writer.close()

if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    run = wandb.init(config=cfg, project='shapes-exp', sync_tensorboard=True)

    main_worker(cfg, args, wandb_run=run)   

    run.finish()
