from tqdm import trange
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from skimage import io, img_as_float32
from tqdm import tqdm
from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import DatasetRepeater


def train(config, generator, discriminator, discriminator2, kp_detector, checkpoint, log_dir, dataset,rank ,device,writer, unsupy, discriminator3):
    train_params = config['train_params']
    num_gpus = torch.cuda.device_count()
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_discriminator2 = torch.optim.Adam(discriminator2.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_discriminator3 = torch.optim.Adam(discriminator3.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, discriminator2,discriminator3, kp_detector,
                                      optimizer_generator, optimizer_discriminator,optimizer_discriminator2,optimizer_discriminator3,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_discriminator2 = MultiStepLR(optimizer_discriminator2, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_discriminator3 = MultiStepLR(optimizer_discriminator3, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    sampler = DistributedSampler(dataset, num_replicas=num_gpus, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'],sampler=sampler, num_workers=32)

    generator_full = GeneratorFullModel(unsupy,kp_detector, generator, discriminator, discriminator2, discriminator3, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator,discriminator2, discriminator3, train_params)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            sampler.set_epoch(epoch)
            total = len(dataloader)
            epoch_train_loss = 0
            generator.train(), discriminator.train(), discriminator2.train(), discriminator3.train(), kp_detector.train()

            with tqdm(total=total) as par:

                for x in dataloader:
                    x['source'] = x['source'].to(device)
                    x['driving'] = x['driving'].to(device)
                    w2 = 0.25
                    w3 = 0.25

                    losses_generator, generated,concat_d_real,concat_d_generated,concat_n_real,concat_n_generated = generator_full(x,w2,w3)

                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    loss.backward()
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()
                    epoch_train_loss+=loss.item()

                    if train_params['loss_weights']['generator_gan'] != 0:
                        optimizer_discriminator.zero_grad()
                        optimizer_discriminator2.zero_grad()
                        optimizer_discriminator3.zero_grad()

                        losses_discriminator = discriminator_full(x, generated,concat_d_real,concat_d_generated,concat_n_real,concat_n_generated,w2,w3)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss = sum(loss_values)

                        loss.backward()
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                        optimizer_discriminator2.step()
                        optimizer_discriminator2.zero_grad()
                        optimizer_discriminator3.step()
                        optimizer_discriminator3.zero_grad()
                    else:
                        losses_discriminator = {}

                    losses_generator.update(losses_discriminator)
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    logger.log_iter(losses=losses)
                    par.update(1)
            
            epoch_train_loss = epoch_train_loss/total
            if (epoch + 1) % train_params['checkpoint_freq'] == 0:
                writer.add_scalar('epoch_train_loss', epoch_train_loss, epoch)
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_discriminator2.step()
            scheduler_discriminator3.step()

            scheduler_kp_detector.step()
            if rank==0:
                logger.log_epoch(epoch, {'generator': generator,
                                        'discriminator': discriminator,
                                        'discriminator2': discriminator2,
                                        'discriminator3': discriminator3,
                                        'kp_detector': kp_detector,
                                        'optimizer_generator': optimizer_generator,
                                        'optimizer_discriminator': optimizer_discriminator,
                                        'optimizer_discriminator2': optimizer_discriminator2,
                                        'optimizer_discriminator3': optimizer_discriminator3,
                                        'optimizer_kp_detector': optimizer_kp_detector} ,inp=x, out=generated)
