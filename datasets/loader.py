#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from .Sampler import DistributedSamplerWrapper
from .new_utlis import worker_init_fn_seed, BalancedBatchSampler
import open_clip
from .build import build_dataset
import open_clip.utils.misc as misc
import numpy as np


def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    labels = [item for sublist in labels for item in sublist]
    inputs, targets, labels, masks = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    targets = [item for sublist in targets for item in sublist]
    masks = [item for sublist in masks for item in sublist]

    inputs, targets, labels, masks = default_collate(inputs), default_collate(targets), default_collate(labels), default_collate(masks)

    return inputs, targets, labels, masks


def construct_loader(cfg, split, transform):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """

    assert split in ["train", "val", "test"]
    data_path = cfg.DATA_LOADER.data_path
    data_name = cfg.TRAIN.DATASET
    shot = cfg.shot
    transform = transform

    normal_json_path = None
    outlier_json_path = None
    if split in ["train"]:
        normal_json_path = cfg.normal_json_path
        outlier_json_path = cfg.outlier_json_path
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    elif split in ["test"]:
        normal_json_path = cfg.val_normal_json_path
        outlier_json_path = cfg.val_outlier_json_path
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    # Construct the dataset
    dataset = build_dataset(data_name, data_path, normal_json_path, outlier_json_path, transform, shot)

    # Create a sampler for multi-process training
    if cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
        collate_func = multiple_samples_collate
    else:
        collate_func = None

    # Create a loader
    if split in ["train"]:
        loader = torch.utils.data.DataLoader(
            dataset,
            worker_init_fn=worker_init_fn_seed,
            batch_sampler = BalancedBatchSampler(cfg, dataset),  # sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            collate_fn=custom_collate_fn
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            collate_fn=custom_collate_fn,
        )

    return loader

# def custom_collate_fn(batch):
#     """
#     Collate function for repeated augmentation and text preprocessing. Each instance in the batch has
#     more than one sample.
#     Args:
#         batch (tuple or list): data batch to collate.
#     Returns:
#         (tuple): collated data batch.
#     """
#     tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')
#     inputs, targets, labels, masks, texts = zip(*batch)

#     labels = torch.tensor([i for i in labels])

#     stacked_inputs = torch.stack([torch.stack(batch) for batch in inputs])
#     transposed_inputs = stacked_inputs.transpose(0, 1)
#     inputs = [tensor for tensor in transposed_inputs]

#     targets = [i for i in targets]
#     masks = torch.stack([i for i in masks])

#     # Process images
#     img = inputs[0].cuda(non_blocking=True)  # img -> (32, 3, 240, 240)
#     normal_image = inputs[1:]
#     normal_image = torch.stack(normal_image)  # normal_image -> (8, 32, 3, 240, 240)
#     shot, b, _, _, _ = normal_image.shape  # shot = 8; b = 32
#     normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)  # normal_image -> (8*32=256, 3, 240, 240)

#     # Process texts
#     pos_texts_list = []
#     neg_texts_list = []

#     for text in texts:
#         obj_type = text.replace('_', " ")
#         normal_texts, anomaly_texts = get_texts(obj_type)

#         pos_tokens = tokenizer(normal_texts)  # pos_tokens -> (154, 77)
#         neg_tokens = tokenizer(anomaly_texts)  # neg_tokens -> (88, 77)

#         pos_texts_list.append(pos_tokens)
#         neg_texts_list.append(neg_tokens)

#     return (img, normal_image, pos_texts_list, neg_texts_list), targets, labels, masks


def custom_collate_fn(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')
    inputs, targets, labels, masks = zip(*batch)

    labels = torch.tensor([i for i in labels])

    stacked_inputs = torch.stack([torch.stack(batch) for batch in inputs])
    transposed_inputs = stacked_inputs.transpose(0, 1)
    inputs = [tensor for tensor in transposed_inputs]

    targets = [i for i in targets]
    
    masks = torch.stack([i for i in masks])

    # inputs, targets, labels, masks = default_collate(inputs), default_collate(targets), default_collate(labels), default_collate(masks)

    return inputs, targets, labels, masks

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the dataset.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler.py type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
