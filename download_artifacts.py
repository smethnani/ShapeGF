import wandb
import argparse
import os

def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='download artifacts')

    parser.add_argument('--save_dir', type=str,
                        help='Log destination')
    parser.add_argument('--run_id', default='',required=True, nargs='+', help="model names")
    parser.add_argument('--artifacts', default='',required=True, nargs='+', help="model names")
    args = parser.parse_args()
    return args

def main():
    path = args.save_dir
    log_dir = os.path.join(path, 'logs')
    run = wandb.init(dir=log_dir, config=cfg, project='ShapeFlow', sync_tensorboard=True, id=args.run_id)
    if not os.path.exists(path):
        os.makedirs(path)
    for artifact_name in args.artifacts:
        artifact = wandb.use_artifact(artifact_name, type='model')
        artifact.download(path)
if __name__ == '__main__':
    args = get_args()
    main(args)