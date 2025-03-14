import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from frames_dataset import FramesDataset
from modules.generator import OcclusionAwareGenerator_SPADE
from modules.discriminator import MultiScaleDiscriminator
from modules.discriminator import MultiScaleDiscriminator2
from modules.discriminator import MultiScaleDiscriminator3
from modules.keypoint_detector import KPDetector
import torch
from train import train
from reconstruction import reconstruction
from animate import animate
import torch.multiprocessing as mp
from modules.unsupmodel import Demo_class


from torch.utils.tensorboard import SummaryWriter 

def main():
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()   
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0,1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args() 
    with open(opt.config) as f:
        config = yaml.full_load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    mp.set_start_method("forkserver") # NCCL does not work with "fork", only with "forkserver" and "spawn"
    local_rank = int(os.environ["LOCAL_RANK"]) 
    print('*****************')
    print(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://') 
    torch.cuda.set_device(local_rank)
    device=torch.device("cuda",local_rank)

    unsupy = Demo_class()
    unsupy.to(device)

    generator = OcclusionAwareGenerator_SPADE(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    generator.to(device)
    if opt.verbose:
        print(generator)
    generator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    discriminator.to(device)
    if opt.verbose:
        print(discriminator)

    discriminator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    discriminator2 = MultiScaleDiscriminator2(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator2.to(device)
    if opt.verbose:
        print(discriminator2)

    discriminator2= torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator2)

    discriminator3 = MultiScaleDiscriminator3(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator3.to(device)
    if opt.verbose:
        print(discriminator3)

    discriminator3= torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator3)


    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    kp_detector.to(device)
    if opt.verbose:
        print(kp_detector)
    kp_detector= torch.nn.SyncBatchNorm.convert_sync_batchnorm(kp_detector)

    kp_detector = DDP(kp_detector,device_ids=[local_rank],broadcast_buffers=False)
    discriminator = DDP(discriminator,device_ids=[local_rank],broadcast_buffers=False)
    discriminator2 = DDP(discriminator2,device_ids=[local_rank],broadcast_buffers=False)
    generator = DDP(generator,device_ids=[local_rank],broadcast_buffers=False)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    writer = SummaryWriter(os.path.join(log_dir,'log'))

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, discriminator2, kp_detector, opt.checkpoint, log_dir, dataset, local_rank, device,writer, unsupy, discriminator3)

    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)

if __name__ == "__main__":
    main()
