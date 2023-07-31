import os
import yaml
import time
import torch.nn as nn
import argparse
import importlib
import wandb
from torch.backends import cudnn
from shutil import copy2
from pprint import pprint


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    parser.add_argument('--log', type=str,
                        help='Log destination')

    # distributed training
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')

    # Resume:
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    # Evaluation split
    parser.add_argument('--eval_split', default='val', type=str,
                        help="The split to be evaluated.")
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

    # #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    logdir = args.log
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    if logdir:
        config.log_name = f"{logdir}/test-{run_time}"
        config.save_dir = f"{logdir}/test-{run_time}"
        config.log_dir = f"{logdir}/test-{run_time}"
    else:
        # Currently save dir and log_dir are the same
        config.log_name = "logs/%s_%s" % (cfg_file_name, run_time)
        config.save_dir = "logs/%s_%s" % (cfg_file_name, run_time)
        config.log_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    os.makedirs(config.log_dir+'/config')
    copy2(args.config, config.log_dir+'/config')
    return args, config


def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True

    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    test_loader = loaders['test_loader']
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    if args.distributed:  # Multiple processes, single GPU per process
        def wrapper(m):
            return nn.DataParallel(m)
        trainer.multi_gpu_wrapper(wrapper)
    trainer.resume(args.pretrained)
    # artifact = wandb.use_artifact('smeth/shapes-exp/epoch_1999_iters_176000.pt:v1', type='model')
    # artifact_dir = artifact.download(root=args.pretrained)
    # trainer.resume(artifact_dir)
    print(cfg.save_dir)
    val_info = trainer.validate(test_loader, epoch=-1)
    # val_info = trainer.test_recon(test_loader, epoch=-1)
    wandb.log(val_info)

    print("Test done:")
    pprint(val_info)


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)
    wandb.init(config=cfg, project='shapes-exp', sync_tensorboard=True)
    main_worker(cfg, args)
    wandb.finish()
