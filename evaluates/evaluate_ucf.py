import json

import torch

from utils.utils import read_labelmap
from evaluates.utils import object_detection_evaluation, standard_fields
import numpy as np
import time
from utils.box_ops import box_iou
import torch
import math


def parse_id(class_num):
    coarse_activity_list_basketball = ['DribbleMove',
                                'DrivesLeft',
                                'DrivesRight',
                                'DrivesStraight',
                                'DriveBaseline',
                                'DriveMiddle',
                                'Post-Up',
                                'Basket',
                                'ToBasket',
                                'FreeThrow',
                                'Jumper',
                                'EarlyJumper',
                                'ToJumper',
                                'TakesEarlyJumpShot',
                                'DribbleJumper',
                                'NoDribbleJumper',
                                'OffensiveRebound',
                                'HighPR',
                                'LeftPR',
                                'RightPR',
                                'HandOff',
                                'BallDelivered',
                                'ToShootersLeft',
                                'ToShootersRight']
    fine2coarse = {
        "Dribble": ["DribbleMove", "DrivesLeft", "DrivesRight", "DrivesStraight", "DriveBaseline", "DriveMiddle"],
        "PostUP": ["Post-Up", ],
        "Basket": ["Basket", "ToBasket"],
        "Jumper": ["FreeThrow", "Jumper", "EarlyJumper", "ToJumper", "TakesEarlyJumpShot", "DribbleJumper", "NoDribbleJumper"],
        "Rebound": ["OffensiveRebound"],
        "PR": ["HighPR", "LeftPR", "RightPR"],
        "Delivered": ["HandOff", "BallDelivered", "ToShootersLeft", "ToShootersRight"]
    }
    
    activity_list_jhmdb = ['brush_hair',
                    'catch',
                    'clap',
                    'climb_stairs',
                    'golf',
                    'jump',
                    'kick_ball',
                    'pick',
                    'pour',
                    'pullup',
                    'push',
                    'run',
                    'shoot_ball',
                    'shoot_bow',
                    'shoot_gun',
                    'sit',
                    'stand',
                    'swing_baseball',
                    'throw',
                    'walk',
                    'wave']
    activity_list_basketball = ['DribbleMove',
                                'DrivesLeft',
                                'DrivesRight',
                                'DrivesStraight',
                                'DriveBaseline',
                                'DriveMiddle',
                                'Post-Up',
                                'Basket',
                                'ToBasket',
                                'FreeThrow',
                                'Jumper',
                                'EarlyJumper',
                                'ToJumper',
                                'TakesEarlyJumpShot',
                                'DribbleJumper',
                                'NoDribbleJumper',
                                'OffensiveRebound',
                                'HighPR',
                                'LeftPR',
                                'RightPR',
                                'HandOff',
                                'BallDelivered',
                                'ToShootersLeft',
                                'ToShootersRight']
    categories = []
    if class_num == 21:
        for i, act_name in enumerate(activity_list_jhmdb):
            categories.append({'id': i + 1, 'name': act_name})
    elif class_num == 24:
        for i, act_name in enumerate(activity_list_basketball):
            categories.append({'id': i + 1, 'name': act_name})
    elif class_num == 7:
        for i, act_name in enumerate(list(fine2coarse.keys())):
            categories.append({'id': i + 1, 'name': act_name})
    return categories


class STDetectionEvaluaterUCF(object):
    '''
    evaluater class designed for multi-iou thresholds
        based on https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py
    parameters:
        dataset that provide GT annos, in the format of AWSCVMotionDataset
        tiou_thresholds: a list of iou thresholds
    attributes:
        clear(): clear detection results, GT is kept
        load_detection_from_path(), load anno from a list of path, in the format of [confi x1 y1 x2 y2 scoresx15]
        evaluate(): run evaluation code
    '''

    def __init__(self, tiou_thresholds=[0.5], load_from_dataset=False, class_num=24):
        categories = parse_id(class_num)
        self.class_num = class_num
        self.categories = categories
        self.tiou_thresholds = tiou_thresholds
        self.lst_pascal_evaluator = []
        self.load_from_dataset = load_from_dataset
        self.exclude_key = []
        for iou in self.tiou_thresholds:
            self.lst_pascal_evaluator.append(
                object_detection_evaluation.PascalDetectionEvaluator(categories, matching_iou_threshold=iou))

    def clear(self):
        for evaluator in self.lst_pascal_evaluator:
            evaluator.clear()

    def load_GT_from_path(self, file_lst):
        # loading data from files
        t_end = time.time()
        sample_dict_per_image = {}
        for path in file_lst:
            data = open(path).readlines()
            for line in data:
                image_key = line.split(' [')[0]
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]
                if (data[4] - data[2]) * (data[5] - data[3]) < 10:
                    self.exclude_key.append(image_key)
                    continue
                scores = np.array(data[6:])
                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }
                # scores = np.max(scores, axis=-1, keepdims=True)

                for x in range(len(scores)):
                    if scores[x] <= 1e-2: continue
                    sample_dict_per_image[image_key]['bbox'].append(
                        np.asarray([data[2], data[3], data[4], data[5]], dtype=float)
                    )
                    sample_dict_per_image[image_key]['labels'].append(x + 1)
                    sample_dict_per_image[image_key]['scores'].append(scores[x])
        # write into evaluator
        for image_key, info in sample_dict_per_image.items():
            if len(info['bbox']) == 0: continue
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_ground_truth_image_info(
                    image_key, {
                        standard_fields.InputDataFields.groundtruth_boxes:
                            np.vstack(info['bbox']),
                        standard_fields.InputDataFields.groundtruth_classes:
                            np.array(info['labels'], dtype=int),
                        standard_fields.InputDataFields.groundtruth_difficult:
                            np.zeros(len(info['bbox']), dtype=bool)
                    })
        print("STDetectionEvaluater: test GT loaded in {:.3f}s".format(time.time() - t_end))

    def load_detection_from_path(self, file_lst):
        # loading data from files
        t_end = time.time()
        sample_dict_per_image = {}

        n = 0
        for path in file_lst:
            print("loading ", path)
            data = open(path).readlines()
            for line in data:
                image_key = line.split(' [')[0]
                if image_key in self.exclude_key:
                    continue
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]

                scores = np.array(data[4:self.class_num + 4])
                if np.argmax(np.array(data[4:])) == len(np.array(data[4:])) - 1:
                    continue

                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }

                x = np.argmax(scores)
                sample_dict_per_image[image_key]['bbox'].append(
                    np.asarray([data[0], data[1], data[2], data[3]], dtype=float)
                )
                sample_dict_per_image[image_key]['labels'].append(x+1)
                sample_dict_per_image[image_key]['scores'].append(scores[x])

        print("start adding into evaluator")
        count = 0
        for image_key, info in sample_dict_per_image.items():
            if len(info['bbox']) == 0:
                print(count)
                continue
            boxes, labels, scores = np.vstack(info['bbox']), np.array(info['labels'], dtype=int), np.array(info['scores'], dtype=float)
            index = np.argsort(-scores)
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_detected_image_info(
                    image_key, {
                        standard_fields.DetectionResultFields.detection_boxes:
                            boxes[index],
                        standard_fields.DetectionResultFields.detection_classes:
                            labels[index],
                        standard_fields.DetectionResultFields.detection_scores:
                            scores[index]
                    })
            count += 1


    def evaluate(self):
        result = {}
        mAP = []
        for x, iou in enumerate(self.tiou_thresholds):
            evaluator = self.lst_pascal_evaluator[x]
            metrics = evaluator.evaluate()
            result.update(metrics)
            mAP.append(metrics['PascalBoxes_Precision/mAP@{}IOU'.format(iou)])
        return mAP, result

