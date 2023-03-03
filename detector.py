import argparse
import torch
import os
from abc import ABC
import numpy as np
from typing import Tuple
from collections import namedtuple


DetectorOutput = namedtuple('DetectorOutput',
                            ['bbox_xywh', 'cls_conf', 'cls_ids'])


class Detector(ABC):
    """
    Abstract class of detector. Calling it with image gives info about detections
    """
    def __init__(self, cli_args: argparse.Namespace or object, *args, **kwargs):
        pass

    def __call__(self, img: np.ndarray, *args, **kwargs) -> DetectorOutput:
        """
        Call detector and get output in form of named tuple "DetectorOutput"
        It consists of 3 fields (N is number of detections):

        'bbox_xywh':    np.array of bounding boxes with shape (N, 4),
                        which has format xywh (i.e. top-left corner and width and height);
        'cls_conf':     class confidences, np.array with shape (N, 1);
        'cls_ids':      class id's, np.array with shape (N, 1);
        """
        pass


class DetectorYolov5(Detector):
    def __init__(self, cli_args, *args, **kwargs):
        super().__init__(cli_args, *args, **kwargs)
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    def __call__(self, img: np.ndarray, *args, **kwargs) -> DetectorOutput:
        """
        Call detector and get output in following form:
        bbox_xywh, cls_conf, cls_ids
        """
        results = self.model(img)
        df = results.pandas().xyxy[0]

        xmin = np.array(df['xmin'])
        xmax = np.array(df['xmax'])
        ymin = np.array(df['ymin'])
        ymax = np.array(df['ymax'])
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        bbox_xywh = np.stack([x, y, w, h]).T

        cls_conf = np.array(df['confidence'])
        cls_ids = np.array(df['class'])

        names = list(df['name'])

        return DetectorOutput(bbox_xywh, cls_conf, cls_ids)





