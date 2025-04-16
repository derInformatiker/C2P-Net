import torch

import regtr.data_loaders.transforms
import regtr.data_loaders.modelnet as modelnet
from regtr.data_loaders.collate_functions import collate_pair
from regtr.data_loaders.threedmatch import ThreeDMatchDataset
from regtr.data_loaders.eardataset import EarDataset, EarDatasetTest

import torchvision


def get_dataloader(cfg, phase, num_workers=0):

    assert phase in ['train', 'val', 'test']

    if cfg.dataset == '3dmatch':
        if phase == 'train':
            # Apply training data augmentation (Pose perturbation and jittering)
            transforms_aug = torchvision.transforms.Compose([
                regtr.data_loaders.transforms.RigidPerturb(perturb_mode=cfg.perturb_pose),
                regtr.data_loaders.transforms.Jitter(scale=cfg.augment_noise),
                regtr.data_loaders.transforms.ShufflePoints(),
                regtr.data_loaders.transforms.RandomSwap(),
            ])
        else:
            transforms_aug = None

        dataset = ThreeDMatchDataset(
            cfg=cfg,
            phase=phase,
            transforms=transforms_aug,
        )
    elif cfg.dataset == 'eardataset':
        if phase == 'train':
            # Apply training data augmentation (Pose perturbation and jittering)
            transforms_aug = torchvision.transforms.Compose([
                regtr.data_loaders.transforms.RigidPerturb(perturb_mode=cfg.perturb_pose),
                regtr.data_loaders.transforms.Jitter(scale=cfg.augment_noise),
                regtr.data_loaders.transforms.ShufflePoints(),
                regtr.data_loaders.transforms.RandomSwap(),
            ])
        else:
            transforms_aug = None

        dataset = EarDataset(
            cfg=cfg,
            phase=phase,
            transforms=transforms_aug,
        )

    elif cfg.dataset == 'eardataset_test':

        dataset = EarDatasetTest(
            cfg=cfg,
            phase=phase,
        )

    elif cfg.dataset == 'modelnet':
        if phase == 'train':
            dataset = regtr.modelnet.get_train_datasets(cfg)[0]
        elif phase == 'val':
            dataset = regtr.modelnet.get_train_datasets(cfg)[1]
        elif phase == 'test':
            dataset = regtr.modelnet.get_test_datasets(cfg)

    else:
        raise AssertionError('Invalid dataset')

    # # For calibrating the number of neighbors (set in config file)
    # from models.backbone_kpconv.kpconv import calibrate_neighbors
    # neighborhood_limits = calibrate_neighbors(dataset, cfg)

    batch_size = cfg[f'{phase}_batch_size']
    shuffle = phase == 'train'

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair,
    )
    return data_loader


