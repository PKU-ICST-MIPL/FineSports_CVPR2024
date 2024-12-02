import glob
import json
import os
import time
import numpy as np

import torch
import math

from .utils import AverageMeter, accuracy, calculate_mAP, read_labelmap
from evaluates.evaluate_ava import STDetectionEvaluater, STDetectionEvaluaterSinglePerson
from evaluates.evaluate_ucf import STDetectionEvaluaterUCF
from apex import amp

# map the color and number to index
color2coloridx = {
    v: idx for idx, v in enumerate([
        "black","white","red","blue","yellow","green"
    ])
}
number2numberidx = {
    v: idx for idx, v in enumerate([
        "23","30","0","35","5","11","6","8","32","34","26","22","3","13","1","2","7","9","10","17","16","55","24","43",
        "4","81","42","41","33","21","25","18","12","36","46","15","14","37","28","44","27","45","99"
    ])
}

def merge_jsons(result_dict, key, output_arr, gt_arr):
    if key not in result_dict.keys():
        result_dict[key] = {"preds": output_arr, "gts": gt_arr}
    else:
        result_dict[key]["preds"] = [max(*l) for l in zip(result_dict[key]["preds"], output_arr)]
        result_dict[key]["gts"] = [max(*l) for l in zip(result_dict[key]["gts"], gt_arr)]

def train_classification(base_iter, model, dataloader, epoch, criterion,
                         optimizer, cfg, writer=None):
    """Task of training video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for step, data in enumerate(dataloader):
        base_iter = base_iter + 1

        train_batch = data[0].cuda()
        train_label = data[1].cuda()
        data_time.update(time.time() - end)
        edd = time.time()
        # print("io: ", edd - end)

        end = edd
        outputs = model(train_batch)
        edd = time.time()
        # print("model: ", edd - end)
        loss = criterion(outputs, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), train_label.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                lr = param['lr']
            print('lr: ', lr)
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            iteration = base_iter
            writer.add_scalar('train_loss_iteration', losses.avg, iteration)
            writer.add_scalar('train_batch_size_iteration', train_label.size(0), iteration)
            writer.add_scalar('learning_rate', lr, iteration)
    return base_iter

def train_postal_detection(cfg, model, criterion, data_loader, optimizer, epoch, max_norm, lr_scheduler, writer=None, model_ema=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()

    device_ = "cuda:" + str(cfg.DDP_CONFIG.GPU)
    loss_cls_text_fuser = torch.nn.CrossEntropyLoss().to(device_).train()
    
    end = time.time()
    model.train()
    criterion.train()


    for idx, data in enumerate(data_loader):
        end = time.time()
        data_time.update(time.time() - end)

        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
        samples = data[0]
        batch_labels = data[-3]

        des = data[-1]
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]
        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)
        image_ids = []
        for t in targets: 
            image_ids.append(t['image_id'])
            del t["image_id"]

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                outputs = model(samples, des, None, batch_labels, 'train')
        if not math.isfinite(outputs["pred_logits"][0].data.cpu().numpy()[0,0]):
            print(outputs["pred_logits"][0].data.cpu().numpy())
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        if epoch > cfg.CONFIG.LOSS_COFS.WEIGHT_CHANGE:
            weight_dict['loss_ce'] = cfg.CONFIG.LOSS_COFS.LOSS_CHANGE_COF

        gt_label = torch.tensor(batch_labels).to(device)
        loss_cls_text = loss_cls_text_fuser(outputs['label_pred'], gt_label)
        
        # gt_color = []
        # gt_number = []
        # for text_ in des:
        #     words = text_.split()
        #     color, number = words[4], words[-1]
        #     gt_color.append(color2coloridx[color])
        #     gt_number.append(number2numberidx[number])
        # gt_color = torch.tensor(gt_color).to(device)
        # gt_number = torch.tensor(gt_number).to(device)
        # loss_cls_color = loss_cls_text_fuser(outputs['color_pred'], gt_color)
        # loss_cls_number = loss_cls_text_fuser(outputs['number_pred'], gt_number)

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) +loss_cls_text

        optimizer.zero_grad()

        losses.backward()
        if max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if cfg.CONFIG.TRAIN.LR_POLICY == 'cosine':
            lr_scheduler.step_update(epoch * len(data_loader) + idx)

        
        
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print_string = 'Train Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
            print(print_string)
            for param in optimizer.param_groups:
                lr = param['lr']
            print('lr: ', lr)

            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)


            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()



            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if cfg.CONFIG.MATCHER.BNY_LOSS:
                losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                exit(1)



            print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f},loss_ce_b: {loss_ce_b:.3f}, loss_text_cls: {loss_cls_text:.3f}'.format(
                class_error=class_err.avg,
                loss=losses_avg.avg,
                loss_bbox=losses_box.avg,
                loss_giou=losses_giou.avg,
                loss_ce=losses_ce.avg,
                loss_ce_b=losses_ce_b.avg,
                loss_cls_text=loss_cls_text.item()
            )
            print(print_string)

            writer.add_scalar('train/class_error', class_err.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/totall_loss', losses_avg.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_bbox', losses_box.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_giou', losses_giou.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_ce', losses_ce.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_ce_b', losses_ce_b.avg, idx + epoch * len(data_loader))

@torch.no_grad()
def validate_postal_detection(cfg, model, criterion, postprocessors, data_loader, epoch, writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()

    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
        if not os.path.exists(tmp_path): os.makedirs(tmp_path)
        tmp_dirs_ = glob.glob("{}/{}/*.txt".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR))
        for tmp_dir in tmp_dirs_:
            os.remove(tmp_dir)
            print("remove {}".format(tmp_dir))
        print("all tmp files removed")

    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)

        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
        samples = data[0]
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]

        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)

        samples = samples.to(device)

        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                outputs = model(samples)

        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)
        for bidx in range(scores.shape[0]):
            frame_id = batch_id[bidx][0]
            key_pos = batch_id[bidx][1]

            if not cfg.CONFIG.MODEL.SINGLE_FRAME:
                out_key_pos = key_pos // cfg.CONFIG.MODEL.DS_RATE

                buff_output.append(scores[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                buff_anno.append(boxes[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                buff_binary.append(output_b[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
            else:
                buff_output.append(scores[bidx])
                buff_anno.append(boxes[bidx])
                buff_binary.append(output_b[bidx])

            for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                buff_id.extend([frame_id])

            raw_idx = (targets[bidx]["raw_boxes"][:, 1] == key_pos).nonzero().squeeze()

            val_label = targets[bidx]["labels"][raw_idx]
            val_label = val_label.reshape(-1, val_label.shape[-1])
            raw_boxes = targets[bidx]["raw_boxes"][raw_idx]
            raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])

            buff_GT_label.append(val_label.detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.detach().cpu().numpy())


            img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                           range(len(raw_boxes))]

            buff_GT_id.extend(img_id_item)

        batch_time.update(time.time() - end)
        end = time.time()

        if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)

            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if cfg.CONFIG.MATCHER.BNY_LOSS:
                losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping eval".format(loss_value))
                print(loss_dict_reduced)
                exit(1)
            print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}, loss_ce_b: {loss_ce_b:.3f}'.format(
                class_error=class_err.avg,
                loss=losses_avg.avg,
                loss_bbox=losses_box.avg,
                loss_giou=losses_giou.avg,
                loss_ce=losses_ce.avg,
                loss_ce_b=losses_ce_b.avg,
            )
            print(print_string)

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        writer.add_scalar('val/class_error', class_err.avg, epoch)
        writer.add_scalar('val/totall_loss', losses_avg.avg, epoch)
        writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
        writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
        writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)
        writer.add_scalar('val/loss_ce_b', losses_ce_b.avg, epoch)

    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    buff_binary = np.concatenate(buff_binary, axis=0)
    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)
    print(buff_output.shape, buff_anno.shape, buff_binary.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))

    tmp_path = '{}/{}/{}.txt'
    with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x], buff_binary[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    Map_ = 0
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        evaluater = STDetectionEvaluater(cfg.CONFIG.DATA.LABEL_PATH, class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print(metrics)
        print_string = 'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        print(mAP)
        writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]

        evaluater = STDetectionEvaluaterSinglePerson(cfg.CONFIG.DATA.LABEL_PATH)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print(metrics)
        print_string = 'person AP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        writer.add_scalar('val/val_person_AP_epoch', mAP[0], epoch)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    time.sleep(30)
    return Map_

@torch.no_grad()
def validate_postal_ucf_detection(cfg, model, criterion, postprocessors, data_loader, epoch, writer, prefix=''):
    print(epoch)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()
    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []
    if True:
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
            if not os.path.exists(tmp_path): 
                os.makedirs(tmp_path)
        for idx, data in enumerate(data_loader):
            data_time.update(time.time() - end)

            device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
            samples = data[0]
            batch_labels = data[-3]

            des = data[-1]
            if cfg.CONFIG.TWO_STREAM:
                samples2 = data[1]
                targets = data[2]
                samples2 = samples2.to(device)
            else:
                targets = data[1]

            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    lfb_features = data[-2]
                    lfb_features = lfb_features.to(device)

                    lfb_location_features = data[-1]
                    lfb_location_features = lfb_location_features.to(device)
                else:
                    lfb_features = data[-1]
                    lfb_features = lfb_features.to(device)

            samples = samples.to(device)
            batch_id = [t["image_id"] for t in targets]

            for t in targets:
                del t["image_id"]

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if cfg.CONFIG.TWO_STREAM:
                if cfg.CONFIG.USE_LFB:
                    if cfg.CONFIG.USE_LOCATION:
                        outputs = model(samples, samples2, lfb_features, lfb_location_features)
                    else:
                        outputs = model(samples, samples2, lfb_features)
                else:
                    outputs = model(samples, samples2)
            else:
                if cfg.CONFIG.USE_LFB:
                    if cfg.CONFIG.USE_LOCATION:
                        outputs = model(samples, lfb_features, lfb_location_features)
                    else:
                        outputs = model(samples, lfb_features)
                else:
                    outputs = model(samples, des, None, batch_labels, 'test')

            loss_dict = criterion(outputs, targets)

            weight_dict = criterion.weight_dict

            orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)
            for bidx in range(scores.shape[0]):

                if len(targets[bidx]["raw_boxes"]) == 0:
                    continue

                frame_id = batch_id[bidx][0]
                key_pos = batch_id[bidx][1]

                out_key_pos = key_pos


                buff_output.append(scores[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                buff_anno.append(boxes[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])

                for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                    buff_id.extend([frame_id])
                    buff_binary.append(output_b[..., 0])

                val_label = targets[bidx]["labels"]
                val_category = torch.full((len(val_label), cfg.CONFIG.DATA.NUM_CLASSES), 0)
                for vl in range(len(val_label)):
                    label = int(val_label[vl])
                    val_category[vl, label] = 1
                val_label = val_category

                raw_boxes = targets[bidx]["raw_boxes"]
                raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])

                buff_GT_label.append(val_label.detach().cpu().numpy())
                buff_GT_anno.append(raw_boxes.detach().cpu().numpy())

                img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                            range(len(raw_boxes))]

                buff_GT_id.extend(img_id_item)

            batch_time.update(time.time() - end)
            end = time.time()

            if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
                print_string = 'Val Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)

                loss_dict_reduced = loss_dict
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                losses_avg.update(loss_value, len(targets))
                losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
                losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
                losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
                class_err.update(loss_dict_reduced['class_error'], len(targets))

                ###
                if cfg.CONFIG.MATCHER.BNY_LOSS:
                    losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))
                ###

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping eval".format(loss_value))
                    print(loss_dict_reduced)
                    exit(1)
                print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}, loss_ce_b: {loss_ce_b:.3f}'.format(
                    class_error=class_err.avg,
                    loss=losses_avg.avg,
                    loss_bbox=losses_box.avg,
                    loss_giou=losses_giou.avg,
                    loss_ce=losses_ce.avg,
                    loss_ce_b=losses_ce_b.avg,
                )
                print(print_string)

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            if prefix == '':
                writer.add_scalar('val/class_error/normal', class_err.avg, epoch)
                writer.add_scalar('val/totall_loss/normal', losses_avg.avg, epoch)
                writer.add_scalar('val/loss_bbox/normal', losses_box.avg, epoch)
                writer.add_scalar('val/loss_giou/normal', losses_giou.avg, epoch)
                writer.add_scalar('val/loss_ce/normal', losses_ce.avg, epoch)
            else:
                try:
                    writer.add_scalar('val/class_error/ema', class_err.avg, epoch)
                    writer.add_scalar('val/totall_loss/ema', losses_avg.avg, epoch)
                    writer.add_scalar('val/loss_bbox/ema', losses_box.avg, epoch)
                    writer.add_scalar('val/loss_giou/ema', losses_giou.avg, epoch)
                    writer.add_scalar('val/loss_ce/ema', losses_ce.avg, epoch)
                except:
                    print('add ema scalar failed')
                    pass

        buff_output = np.concatenate(buff_output, axis=0)
        buff_anno = np.concatenate(buff_anno, axis=0)
        buff_binary = np.concatenate(buff_binary, axis=0)

        buff_GT_label = np.concatenate(buff_GT_label, axis=0)
        buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)

        print(buff_output.shape, buff_anno.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))

        tmp_path = '{}/{}/gpu_{}_epoch_{}_{}.txt'
        with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK, epoch, prefix), 'w') as f:
            for x in range(len(buff_id)):
                data = np.concatenate([buff_anno[x], buff_output[x]])
                f.write("{} {}\n".format(buff_id[x], data.tolist()))

        tmp_binary_path = '{}/{}/binary_gpu_{}_epoch_{}_{}.txt'
        with open(tmp_binary_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK, epoch, prefix), 'w') as f:
            for x in range(len(buff_id)):
                data = buff_binary[x]
                f.write("{} {}\n".format(buff_id[x], data.tolist()))

        tmp_GT_path = '{}/{}/GT_gpu_{}_epoch_{}_{}.txt'
        with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK, epoch, prefix), 'w') as f:
            for x in range(len(buff_GT_id)):
                data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
                f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    if cfg.DDP_CONFIG.DISTRIBUTED:
        torch.distributed.barrier()
    Map_ = 0
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        evaluater = STDetectionEvaluaterUCF(class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR,x, epoch, prefix) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x,epoch, prefix) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print(metrics)
        print_string = prefix+'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        print(mAP)
        if prefix == '':
            writer.add_scalar('val/val_mAP_epoch/normal', mAP[0], epoch)
        else:
            try:
                writer.add_scalar('val/val_mAP_epoch/ema', mAP[0], epoch)
            except:
                pass
        Map_ = mAP[0]
    if cfg.DDP_CONFIG.DISTRIBUTED:
        torch.distributed.barrier()
    time.sleep(30)
    return Map_


@torch.no_grad()
def validate_postal_ucf_detection_ema(cfg, model, criterion, postprocessors, data_loader, epoch, writer, prefix=''):
    print(epoch)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()

    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []
    if True:
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
            if not os.path.exists(tmp_path): 
                os.makedirs(tmp_path)

        for idx, data in enumerate(data_loader):
            data_time.update(time.time() - end)


            device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
            samples = data[0]
            batch_labels = data[-3]

            des = data[-1]
            if cfg.CONFIG.TWO_STREAM:
                samples2 = data[1]
                targets = data[2]
                samples2 = samples2.to(device)
            else:
                targets = data[1]

            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    lfb_features = data[-2]
                    lfb_features = lfb_features.to(device)

                    lfb_location_features = data[-1]
                    lfb_location_features = lfb_location_features.to(device)
                else:
                    lfb_features = data[-1]
                    lfb_features = lfb_features.to(device)

            samples = samples.to(device)

            batch_id = [t["image_id"] for t in targets]

            for t in targets:
                del t["image_id"]

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if cfg.CONFIG.TWO_STREAM:
                if cfg.CONFIG.USE_LFB:
                    if cfg.CONFIG.USE_LOCATION:
                        outputs = model(samples, samples2, lfb_features, lfb_location_features)
                    else:
                        outputs = model(samples, samples2, lfb_features)
                else:
                    outputs = model(samples, samples2)
            else:
                if cfg.CONFIG.USE_LFB:
                    if cfg.CONFIG.USE_LOCATION:
                        outputs = model(samples, lfb_features, lfb_location_features)
                    else:
                        outputs = model(samples, lfb_features)
                else:
                    outputs = model(samples, des, None, batch_labels)

            loss_dict = criterion(outputs, targets)

            weight_dict = criterion.weight_dict

            orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)
            for bidx in range(scores.shape[0]):

                if len(targets[bidx]["raw_boxes"]) == 0:
                    continue

                frame_id = batch_id[bidx][0]
                key_pos = batch_id[bidx][1]

                out_key_pos = key_pos

                buff_output.append(scores[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                buff_anno.append(boxes[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])

                for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                    buff_id.extend([frame_id])
                    buff_binary.append(output_b[..., 0])

                val_label = targets[bidx]["labels"]
                val_category = torch.full((len(val_label), cfg.CONFIG.DATA.NUM_CLASSES), 0)
                for vl in range(len(val_label)):
                    label = int(val_label[vl])
                    val_category[vl, label] = 1
                val_label = val_category

                raw_boxes = targets[bidx]["raw_boxes"]
                raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])

                buff_GT_label.append(val_label.detach().cpu().numpy())
                buff_GT_anno.append(raw_boxes.detach().cpu().numpy())

                img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                            range(len(raw_boxes))]

                buff_GT_id.extend(img_id_item)

            batch_time.update(time.time() - end)
            end = time.time()

            if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
                print_string = 'Val Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)

                loss_dict_reduced = loss_dict
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                losses_avg.update(loss_value, len(targets))
                losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
                losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
                losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
                class_err.update(loss_dict_reduced['class_error'], len(targets))

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping eval".format(loss_value))
                    print(loss_dict_reduced)
                    exit(1)
                print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}'.format(
                    class_error=class_err.avg,
                    loss=losses_avg.avg,
                    loss_bbox=losses_box.avg,
                    loss_giou=losses_giou.avg,
                    loss_ce=losses_ce.avg
                )
                print(print_string)

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            if prefix == '':
                writer.add_scalar('val/class_error/normal', class_err.avg, epoch)
                writer.add_scalar('val/totall_loss/normal', losses_avg.avg, epoch)
                writer.add_scalar('val/loss_bbox/normal', losses_box.avg, epoch)
                writer.add_scalar('val/loss_giou/normal', losses_giou.avg, epoch)
                writer.add_scalar('val/loss_ce/normal', losses_ce.avg, epoch)
            else:
                try:
                    writer.add_scalar('val/class_error/ema', class_err.avg, epoch)
                    writer.add_scalar('val/totall_loss/ema', losses_avg.avg, epoch)
                    writer.add_scalar('val/loss_bbox/ema', losses_box.avg, epoch)
                    writer.add_scalar('val/loss_giou/ema', losses_giou.avg, epoch)
                    writer.add_scalar('val/loss_ce/ema', losses_ce.avg, epoch)
                except:
                    print('add ema scalar failed')
                    pass

        buff_output = np.concatenate(buff_output, axis=0)
        buff_anno = np.concatenate(buff_anno, axis=0)
        buff_binary = np.concatenate(buff_binary, axis=0)

        buff_GT_label = np.concatenate(buff_GT_label, axis=0)
        buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)

        print(buff_output.shape, buff_anno.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))

        tmp_path = '{}/{}/gpu_{}_epoch_{}_{}.txt'
        with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK, epoch, prefix), 'w') as f:
            for x in range(len(buff_id)):
                data = np.concatenate([buff_anno[x], buff_output[x]])
                f.write("{} {}\n".format(buff_id[x], data.tolist()))

        tmp_binary_path = '{}/{}/binary_gpu_{}_epoch_{}_{}.txt'
        with open(tmp_binary_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK, epoch, prefix), 'w') as f:
            for x in range(len(buff_id)):
                data = buff_binary[x]
                f.write("{} {}\n".format(buff_id[x], data.tolist()))

        tmp_GT_path = '{}/{}/GT_gpu_{}_epoch_{}_{}.txt'
        with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK, epoch, prefix), 'w') as f:
            for x in range(len(buff_GT_id)):
                data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
                f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    if cfg.DDP_CONFIG.DISTRIBUTED:
        torch.distributed.barrier()
    Map_ = 0
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        evaluater = STDetectionEvaluaterUCF(class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR,x, epoch, prefix) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x,epoch, prefix) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print(metrics)
        print_string = prefix+'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        print(mAP)
        if prefix == '':
            writer.add_scalar('val/val_mAP_epoch/normal', mAP[0], epoch)
        else:
            try:
                writer.add_scalar('val/val_mAP_epoch/ema', mAP[0], epoch)
            except:
                pass
        Map_ = mAP[0]
    if cfg.DDP_CONFIG.DISTRIBUTED:
        torch.distributed.barrier()
    time.sleep(30)
    return Map_