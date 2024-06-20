# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


# python test.py --val_normal_json_path /AD_json/bottle_val_normal.json --val_outlier_json_path /AD_json/bottle_val_outlier.json --category bottle --few_shot_dir /fs_samples/visa/2/

"""Wrapper to train/test models."""

import argparse
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"                                                                                        # check
import torch

from engine_test import test
from open_clip.utils.misc import launch_job
import open_clip.utils.checkpoint as cu
from open_clip.config.defaults import assert_and_infer_cfg, get_cfg
shot_number = 8


def parse_args(val_normal_json,val_outlier_json,dir_path, obj, fs_samples):
    """
    Parse the following arguments for a default parser.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:8888",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See mvit/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--model",
        help="model_name",
        default="ViT-B-16-plus-240",
        type=str,
    )
    parser.add_argument(
        "--pretrained",
        help="whether use pretarined model",
        default=None,
        type=str
    )
    parser.add_argument('--normal_json_path', default='./datasets/AD_json/hyperkvasir_normal.json', nargs='+', type=str,
                        help='json path')
    parser.add_argument('--outlier_json_path', default='./datasets/AD_json/hyperkvasir_outlier.json', nargs='+', type=str,
                        help='json path')
    parser.add_argument('--val_normal_json_path', default=val_normal_json, nargs='+', type=str,
                        help='json path')
    parser.add_argument('--val_outlier_json_path', default=val_outlier_json, nargs='+', type=str,
                        help='json path')
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="the number of batches per epoch")
    parser.add_argument("--dataset_dir", type=str, default="./", help="the number of batches per epoch")
    parser.add_argument("--category", type=str, default=obj, help="")                                                               # check
    parser.add_argument(
        "--shot", type=int, default=shot_number, help="size for visual prompts"                                                               # check
    )
    parser.add_argument("--image_size", type=int, default=240, help="image size")
    parser.add_argument("--few_shot_dir", type=str, default=os.path.join(fs_samples,str(shot_number)), help="path to few shot sample prompts")                 # check

    # if len(sys.argv) == 1:
    #     parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "normal_json_path"):
        cfg.normal_json_path = args.normal_json_path
    if hasattr(args, "outlier_json_path"):
        cfg.outlier_json_path = args.outlier_json_path
    if hasattr(args, "val_normal_json_path"):
        cfg.val_normal_json_path = args.val_normal_json_path
    if hasattr(args, "val_outlier_json_path"):
        cfg.val_outlier_json_path = args.val_outlier_json_path
    if hasattr(args, "steps_per_epoch"):
        cfg.steps_per_epoch = args.steps_per_epoch
    if hasattr(args, "dataset_dir"):
        cfg.dataset_dir = args.dataset_dir
    if hasattr(args, "category"):
        cfg.category = args.category

    if hasattr(args, "local_rank"):
        cfg.local_rank = args.local_rank

    if hasattr(args, "model"):
        cfg.model = args.model

    if hasattr(args, "pretrained"):
        cfg.pretrained = args.pretrained

    if hasattr(args, "shot"):
        cfg.shot = args.shot

    if hasattr(args, "image_size"):
        cfg.image_size = args.image_size

    if hasattr(args, "few_shot_dir"):
        cfg.few_shot_dir = args.few_shot_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def main(dir_path, fs_samples):
    """
    Main function to spawn the train and test process.
    """

    object_dict = {}
    for json_file in os.listdir(dir_path):
        if json_file.endswith("_val_normal.json"):
            key = json_file[:-len("_val_normal.json")]
            if key not in object_dict:
                object_dict[key] = {"normal": []}
            elif "normal" not in object_dict[key]:
                object_dict[key]["normal"] = []
            object_dict[key]["normal"].append(os.path.join(dir_path, json_file))
        elif json_file.endswith("_val_outlier.json"):
            key = json_file[:-len("_val_outlier.json")]
            if key not in object_dict:
                object_dict[key] = {"outlier": []}
            elif "outlier" not in object_dict[key]:
                object_dict[key]["outlier"] = []
            object_dict[key]["outlier"].append(os.path.join(dir_path, json_file))
    for obj, dict_of_list in object_dict.items():
        print("Object is : ", obj)
        args = parse_args(object_dict[obj]["normal"],object_dict[obj]["outlier"],dir_path, obj, fs_samples)
        cfg = load_config(args)
        cfg = assert_and_infer_cfg(cfg)

        # Perform testing.
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    print("shot number is :", shot_number)
    dir_paths = {                                                                                                              # check
                # "./AD_json/" : "",
                #  "/home/medical/Anomaly_Project/InCTRL/AD_json/visa/" : "/home/medical/Anomaly_Project/InCTRL/data/fs_samples/visa/",
                 "/home/medical/Anomaly_Project/InCTRL/AD_json/brainmri/" : "/home/medical/Anomaly_Project/InCTRL/data/fs_samples/BrainMRI/",
                #  "/home/medical/Anomaly_Project/InCTRL/AD_json/headct/" : "/home/medical/Anomaly_Project/InCTRL/data/fs_samples/HeadCT/",
                #  "/home/medical/Anomaly_Project/InCTRL/AD_json/sdd/" : "/home/medical/Anomaly_Project/InCTRL/SDD/SDD/",
                #  "/home/medical/Anomaly_Project/InCTRL/AD_json/elpv/" : "/home/medical/Anomaly_Project/InCTRL/elpv",
                #  "/home/medical/Anomaly_Project/InCTRL/AD_json/aitex/": "/home/medical/Anomaly_Project/InCTRL/AITEX"
    }
    for dir_path, fs_samples in dir_paths.items():
        print(dir_path)
        main(dir_path, fs_samples)

