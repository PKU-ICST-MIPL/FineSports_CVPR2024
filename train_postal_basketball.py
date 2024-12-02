import argparse
import os
import time
import torch
import torch.optim
from tensorboardX import SummaryWriter

from models.postal_basketball import build_model
from utils.model_utils import deploy_model, load_model, save_model
from utils.video_action_recognition import validate_postal_ucf_detection, train_postal_detection, validate_postal_ucf_detection_ema
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir
from models.ema_model import ExponentialMovingAverage
from datasets.basketball_jhmdb_frame import build_dataloader
from utils.lr_scheduler import build_scheduler
from apex import amp


def main_worker(cfg):

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None


    print('Creating PoSTAL model: %s' % cfg.CONFIG.MODEL.NAME)
    model, criterion, postprocessors = build_model(cfg)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and "class_embed" not in n and "query_embed" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model.named_parameters() if "class_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR,
        },
        {
            "params": [p for n, p in model.named_parameters() if "query_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR,
        },
    ]

    train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_dataloader(cfg)

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.CONFIG.TRAIN.LR, weight_decay=cfg.CONFIG.TRAIN.W_DECAY)
    if cfg.CONFIG.TRAIN.LR_POLICY == "step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.CONFIG.TRAIN.LR_MILESTONE, gamma=0.1)
    else:
        lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    if cfg.CONFIG.MODEL.LOAD:
        model, cfg.CONFIG.TRAIN.START_EPOCH = load_model(model, cfg, optimizer, lr_scheduler, load_fc=cfg.CONFIG.MODEL.LOAD_FC)
    torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
    model.cuda()

    
    model = deploy_model(model, cfg)
    model_wo_ddp = model.module
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))
    print("test sampler", len(val_loader))

    criterion = criterion.cuda()

    model_ema = None
    if cfg.CONFIG.MODEL.USE_EMA:
        print('****** using ema model ******')
        print('ema step: ', cfg.CONFIG.MODEL.EMA_STEP)
        adjust = cfg.DDP_CONFIG.GPU_WORLD_SIZE*cfg.CONFIG.TRAIN.BATCH_SIZE* cfg.CONFIG.MODEL.EMA_STEP *cfg.CONFIG.TRAIN.EPOCH_NUM
        alpha = 1.0 - 0.99998
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model_wo_ddp, device="cpu" ,decay=0.99)
    
    print('Start training...')

    for epoch in range(cfg.CONFIG.TRAIN.START_EPOCH, cfg.CONFIG.TRAIN.EPOCH_NUM):
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)
        time.sleep(1)
        train_postal_detection(cfg, model, criterion, train_loader, optimizer, epoch, 
                              cfg.CONFIG.LOSS_COFS.CLIPS_MAX_NORM, lr_scheduler, writer, model_ema=model_ema)
        
        if epoch % cfg.CONFIG.VAL.FREQ == 0:
            if model_ema is not None:
                model_ema.to("cuda:"+str(cfg.DDP_CONFIG.GPU))
                validate_postal_ucf_detection_ema(cfg, model_ema, criterion, postprocessors, val_loader, epoch, writer, prefix='EMA: ')
                model_ema.to('cpu')
            validate_postal_ucf_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer, prefix='')
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            save_model(model, optimizer, lr_scheduler, epoch, cfg)

        lr_scheduler.step(epoch) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='configuration/basketball.yaml',
                        help='path to config file.')
    parser.add_argument('--ident',
                        default='./save_pth',
                        help='path to config file.')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.ident = args.ident
    cfg.CONFIG.LOG.RES_DIR = args.ident+'/tmp'
    spawn_workers(main_worker, cfg)
