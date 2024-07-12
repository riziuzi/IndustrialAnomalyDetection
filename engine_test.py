# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import random
import json
import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
import open_clip.utils.checkpoint as cu
import open_clip.utils.distributed as du
import open_clip.utils.logging as logging
import open_clip.utils.misc as misc
import numpy as np
import torch
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
from binary_focal_loss import BinaryFocalLoss
import torch.distributed as dist
import matplotlib.pyplot as plt
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from PIL import Image
from tqdm import tqdm
import progressbar
from image_display import display_image, create_prediction_image
import seaborn as sns

logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')
def peek_image(image_tensor):

        plt.figure(figsize=(10, 10))
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Transpose to HWC format
        plt.imshow(image_np)
        plt.title("Corresponding Image")
        plt.axis('off')
        plt.savefig("corresponding_image.png", dpi=300, bbox_inches='tight')
        plt.close()
@torch.no_grad()
def eval_epoch(val_loader, model, cfg, tokenizer, normal_list, mode=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    total_label = torch.Tensor([]).cuda()
    total_pred = torch.Tensor([]).cuda()
    total_iterations = len(val_loader)
    progress_bar = progressbar.ProgressBar(max_value=total_iterations, prefix="Validation")
    for cur_iter, (inputs, types, labels) in enumerate(val_loader):
        print("Entered the iteration loop")
        # print("Validation iteration : ", cur_iter, f"/{len(val_loader)}")
        # if cur_iter<=-1: continue
        if cfg.NUM_GPUS:
            labels = labels.cuda()
        # peek_image(inputs[0][-2])
        # peek_image(inputs[0][0])
        # inputs[0] = inputs[0].flip(0)
        # peek_image(inputs[0][-2])
        # peek_image(inputs[0][0])
        # labels.flip(0)
        preds, _ = model(tokenizer, inputs, types, normal_list)

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)
        # progress_bar.update(cur_iter + 1)

        total_pred = total_pred.cpu().numpy() 
        total_label = total_label.cpu().numpy()
        
        # create_prediction_visualization(total_label, total_pred)
        print("Predict " + mode + " set: ")
        try:
            total_roc, total_pr = aucPerformance(total_pred, total_label)
        except:
            pass
        total_pred = torch.tensor(total_pred).cuda()
        total_label = torch.tensor(total_label).cuda()
        print("Exited the iteration loop")
    # break
    create_prediction_image(total_label, total_pred)
    return total_roc

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;


def drawing(cfg, data, xlabel, ylabel, dir):
    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(data, 'b', label='loss')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, dir))


def test(cfg, load=None, mode = None):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    device = torch.cuda.current_device()

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = open_clip.model.InCTRL(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    model = model.cuda(device=device)

    cu.load_test_checkpoint(cfg, model)
    
    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    if load == None:
        load = loader.construct_loader(cfg, "test", transform)
        mode = "test"

    few_shot_path = os.path.join(cfg.few_shot_dir, cfg.category+".pt")
    normal_list = torch.load(few_shot_path)
    display_image(normal_list)
    # Create meters.
    total_roc = eval_epoch(load, model, cfg, tokenizer, normal_list, mode)

    return total_roc
