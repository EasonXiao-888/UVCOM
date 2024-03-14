import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from uvcom.config import BaseOptions
from uvcom.start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from uvcom.start_end_dataset_audio import \
    StartEndDataset_audio, start_end_collate_audio, prepare_batch_inputs_audio
from uvcom.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.use_deterministic_algorithms(True)

def eval_(model, criterion, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "[Epoch] {epoch:03d} [lr] {lr} [Loss] {loss_str}\n"
    # opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    opt.eval_log_txt_formatter = "[Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, val_dataset, opt, save_submission_filename, None, criterion, tb_writer)
    logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))

def start_eval():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    print('##################')
    print(opt.a_feat_dir is None)
    print(opt.a_feat_dir)
    print('##################')
    if opt.a_feat_dir is None:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset(**dataset_config)
    else:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            a_feat_dir=opt.a_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset_audio(**dataset_config)



    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        if opt.a_feat_dir is None:
            eval_dataset = StartEndDataset(**dataset_config)
        else:
            eval_dataset = StartEndDataset_audio(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    
    eval_(model, criterion, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_eval()
    # if not debug:
    #     input_args = ["--resume", best_ckpt_path,
    #                   "--eval_split_name", eval_split_name,
    #                   "--eval_path", eval_path]

    #     import sys
    #     sys.argv[1:] = input_args
    #     logger.info("\n\n\nFINISHED TRAINING!!!")
    #     logger.info("Evaluating model at {}".format(best_ckpt_path))
    #     logger.info("Input args {}".format(sys.argv[1:]))
    #     start_inference(opt)
