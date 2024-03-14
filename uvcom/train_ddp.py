import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import misc_ddp as utils
from torch.nn.parallel import DistributedDataParallel as DDP

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

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12357'

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


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples,
                                 disable= not utils.is_main_process()):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        if opt.a_feat_dir is None:
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        else:
            model_inputs, targets = prepare_batch_inputs_audio(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()
        model_inputs['epoch'] = epoch_i
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    # print/add logs
    if opt.main_process:
        tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

        lr = optimizer.param_groups[0]["lr"]

        to_write = opt.train_log_txt_formatter.format(
        # time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i+1,
            lr=lr,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)

        logger.info("Epoch time stats:")
        for name, meter in time_meters.items():
            d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
            logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    # if opt.device.type == "cuda":
    #     logger.info("CUDA enabled.")
    #     model.to(opt.device)

    if opt.main_process:
        tb_writer = SummaryWriter(opt.tensorboard_log_dir)
        tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
        opt.train_log_txt_formatter = "[Epoch] {epoch:03d} [lr] {lr} [Loss] {loss_str}\n"
        # opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
        opt.eval_log_txt_formatter = "[Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    else:
        tb_writer = None

    if opt.a_feat_dir is None:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=False,
            pin_memory=opt.pin_memory,
            sampler=DistributedSampler(train_dataset,num_replicas=opt.world_size, rank=opt.rank,
                                                    shuffle=True, seed=opt.seed, drop_last=False)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate_audio,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=False,
            pin_memory=opt.pin_memory,
            sampler=DistributedSampler(train_dataset,num_replicas=opt.world_size, rank=opt.rank,
                                                    shuffle=True, seed=opt.seed, drop_last=False)
        )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch",disable=not opt.main_process):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.main_process and opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                # time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            if opt.main_process:
                with open(opt.eval_log_filepath, "a") as f:
                    f.write(to_write)
                logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
                if metrics_nms is not None:
                    logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

                metrics = metrics_no_nms
                for k, v in metrics["brief"].items():
                    tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            if 'charades' in opt.dset_name:
                stop_score = metrics["brief"]["MR-full-R1@0.7"] + metrics["brief"]["MR-full-R1@0.5"]
            else:
                stop_score = metrics["brief"]["MR-full-mAP"]
                # stop_score = metrics["brief"]["HL-min-VeryGood-mAP"]  # for QV_ablation only for hl
                
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        # save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        # if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
        #     checkpoint = {
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch_i,
        #         "opt": opt
        #     }
        #     torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break
        
    if opt.main_process in [0, -1]:
        tb_writer.close()



def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "[Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "[Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            # stop_score = metrics["brief"]["MR-full-mAP"]
            stop_score = metrics["brief"]["mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        # save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        # if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
        #     checkpoint = {
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch_i,
        #         "opt": opt
        #     }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()

def init_process_group_and_set_device(world_size, process_id, device_id, config):
    """
    This function needs to be called on each spawned process to initiate learning using DistributedDataParallel.
    The function initiates the process' process group and assigns it a single GPU to use during training.
    """
    config.world_size = world_size
    config.rank = process_id
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    config.device = device
    if world_size > 1:
        config.distributed = True
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        utils.setup_for_distributed(config.rank == 0)
    else:
        config.distributed = False
    return device


def start_training(process_id,args):
    opt = args
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    opt.main_process = process_id == 0
    # import pdb;pdb.set_trace()
    device = init_process_group_and_set_device(args.num_devices, process_id, args.device_ids[process_id], args)

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)

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

    model.to(device)
    logger.info(f"Using {torch.cuda.device_count()} GPUs.")
    model = DDP(model,
                device_ids=[args.device_ids[process_id]])
                                                    #       output_device=local_rank,
                                                    #   find_unused_parameters=True)
    if opt.main_process :
        logger.info(f"Model {model}")
        count_parameters(model)
        logger.info("Start Training...")
    
    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    args = BaseOptions().parse()
    if hasattr(args, 'num_gpus'):
        args.num_devices = max(min(args.num_gpus, torch.cuda.device_count()), 1)
        args.device_ids = list(range(args.num_gpus))
    torch.multiprocessing.spawn(start_training, nprocs=args.num_devices, args=(args,))
    # best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
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
