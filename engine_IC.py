# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import re
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
# from torchsummary import summary
from torchsummary import summary
from torch.nn.modules.module import _addindent
import progressbar
import matplotlib.pyplot as plt
from image_display import display_image
import torch
from torch.utils.tensorboard import SummaryWriter
logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')

def train_epoch(
    train_loader,
    model,
    optimizer,
    tokenizer,
    cfg
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """
    # Enable train mode.
    model.train()
    widgets = [
        ' [', progressbar.Percentage(), '] ',
        progressbar.Bar(), ' (', progressbar.ETA(), ') ',
    ]
    pbar = progressbar.ProgressBar(max_value=len(train_loader), widgets=widgets)

    all_loss = 0.0
    for cur_iter, (inputs, types, labels, masks) in enumerate(train_loader):

        if cfg.NUM_GPUS:
            labels = labels.cuda()
            masks = masks.cuda()
        
        
        preds, preds2, preds3 = model(inputs, None)
        loss_fun = BinaryFocalLoss()
        loss_fun = loss_fun.cuda()

        # Compute the loss.
        # loss = loss_fun(preds, labels.float()) + loss_fun(preds2, labels.float())
        loss = loss_fun(preds3, masks.float())

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        # dist.all_reduce(loss)
        loss_value = loss.item()
        all_loss = all_loss + loss_value
        pbar.update(cur_iter + 1)

    all_loss = all_loss / (cur_iter + 1)
    print("train_loss: ", all_loss)
    return all_loss


@torch.no_grad()
def eval_epoch(val_loader, model, cfg, tokenizer, mode=None):
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
    widgets = [
        ' [', progressbar.Percentage(), '] ',
        progressbar.Bar(), ' (', progressbar.ETA(), ') ',
    ]
    pbar = progressbar.ProgressBar(max_value=len(val_loader), widgets=widgets)
    for cur_iter, (inputs, types, labels) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            labels = labels.cuda()
        display_image(inputs[0])
        preds, _ = model(tokenizer, inputs, types, None)

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)
        pbar.update(cur_iter+1)

    total_pred = total_pred.cpu().numpy()  #.squeeze()
    total_label = total_label.cpu().numpy()

    print("Predict " + mode + " set: ")
    total_roc, total_pr = aucPerformance(total_pred, total_label)

    return total_roc, total_pr

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.NUM_GPUS:
        device = torch.cuda.current_device()

    # Build the model and print model statistics.
    # cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    cf = '/home/medical/Anomaly_Project/InCTRL/open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = open_clip.model.InCTRL(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    # input_data = [torch.randn(16, 3, 224, 224) for i in range(5)]
    # summary(model, input_data=input_data)
    # print(torch_summarize(model))

    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        # Transfer the model to the current GPU device
        model = model.cuda(device=device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[device], output_device=device
        )

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # checkpoint_path = "./vit_b_16_plus_240-laion400m_e32-699c4b84.pt"
    # checkpoint_dir = "./tmp/checkpoints/"
    # checkpoint_dir = "/home/medical/Anomaly_Project/InCTRL/TEMP2checkpoints/checkpoints/8/"
    checkpoint_dir = "/home/medical/Anomaly_Project/InCTRL/ORIGINALcheckpoints/checkpoints/8/"
    start_epoch = load_latest_checkpoint(model, checkpoint_dir)
    start_epoch = 0                                                         # since for each training dataset, model should train from start
    # if os.path.exists(checkpoint_path):
    #     # Load the checkpoint
    #     with pathmgr.open(checkpoint_path, "rb") as f:
    #         checkpoint = torch.load(f, map_location="cpu")
        
    #     # Retrieve necessary information from the checkpoint
    #     model_state_dict = checkpoint['model_state_dict']
    #     if(checkpoint['epoch']):
    #         start_epoch = checkpoint['epoch']
    #     else:
    #         start_epoch = 0
    #     # Load the model's state dictionary
    #     model.load_state_dict(model_state_dict)
        
    #     print("Checkpoint loaded successfully.")
    # else:
    #     start_epoch = 0
    #     print("Checkpoint file does not exist. Starting training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=[0.9, 0.999])

    # Create the train and val loaders.

    print("Loading Train/Test Loaders ...")
    train_loader = loader.construct_loader(cfg, "train", transform)
    # test_loader = loader.construct_loader(cfg, "test", transform)
    print("Loading Complete!")

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    epoch_losses = []
    test_losses = []
    epoch_timer = EpochTimer()
    writer = SummaryWriter()
    for cur_epoch in range(start_epoch, 1):
        print("Epoch: ", cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        epoch_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            tokenizer,
            cfg,
        )
        writer.add_scalar("Loss/train", epoch_loss, cur_epoch)
        epoch_losses.append(epoch_loss)
        epoch_timer.epoch_toc()
        # logger.info(
        #     f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
        #     f"from {start_epoch} to {cur_epoch} take "
        #     f"{epoch_timer.avg_epoch_time():.2f}s in average and "
        #     f"{epoch_timer.median_epoch_time():.2f}s in median."
        # )
        # logger.info(
        #     f"For epoch {cur_epoch}, each iteraction takes "
        #     f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
        #     f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
        #     f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        # )

        
        save_checkpoint(model, checkpoint_dir)
        print(f"Time taken for epoch number {cur_epoch} -> {epoch_timer.avg_epoch_time():.2f}s in average and {epoch_timer.median_epoch_time():.2f}s in median.")
        print(f"epoch loss : {epoch_loss}")
        print("\n")
        print("\n")
        # total = eval_epoch(train_loader, model, cfg, tokenizer, "train")
        # test = eval_epoch(test_loader, model, cfg, tokenizer, "test")
        # test_losses.append(test)
        # print("Total ROC and PR: ", total, "AND Test ROC and PR : ", test)
        # print("Test ROC and PR : ", test)
    print("End fo Epochs")
    print(f"List of Epoch losses -> {epoch_losses}")
    # print(f"List of Test losses (ROC, PR) -> {test_losses}")
    writer.flush()
    print("\n")
    print("\n")
    print("\n")


def save_checkpoint(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path,"checkpoint.pyth"))
    print(f"Checkpoint saved to {path}")


def load_latest_checkpoint(model, checkpoint_dir):
    # try:
        # List all checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pyth') or f.endswith('.pt')]
        
        if not checkpoint_files:
            print("No checkpoint files found. Starting training from scratch.")
            return 0

        # # Extract epoch numbers from checkpoint file names
        # epoch_numbers = []
        # for file_name in checkpoint_files:
        #     match = re.search(r'checkpoint_(\d+)\.(pyth|pt)', file_name)
        #     if match:
        #         epoch_number = int(match.group(1))
        #         epoch_numbers.append((epoch_number, file_name))

        # Find the checkpoint with the highest epoch number
        if os.path.exists(os.path.join(checkpoint_dir,"checkpoint.pyth")):
            # latest_epoch, latest_checkpoint_file = max(epoch_numbers, key=lambda x: x[0])
            # latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            
            checkpoint = torch.load(os.path.join(checkpoint_dir,"checkpoint.pyth"), map_location="cpu")
            # model_state_dict = model.state_dict()
            # filtered_state_dict = {k: v for k,v in checkpoint.items if k in model_state_dict}                       # things which is in both checkpoint as well as new model
            # model_state_dict.update(filtered_state_dict)
            model.load_state_dict(checkpoint, strict=False)
            # latest_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{os.path.join(checkpoint_dir, 'checkpoint.pyth')}'")

            # print(f"Loaded checkpoint '{os.path.join(checkpoint_dir,"model.pt")}' (epoch {latest_epoch})")
            return 0
        else:
            print("No valid checkpoint files found. Starting training from scratch.")
            return 0
    # except:
    #     print("State not loaded!")


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

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
    # print("Skipping loading model wegihts")
    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    if load == None:
        load = loader.construct_loader(cfg, "test", transform)
        mode = "test"

    # Create meters.
    total_roc = eval_epoch(load, model, cfg, tokenizer, mode)

    return total_roc
