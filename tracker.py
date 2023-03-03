import argparse
from abc import ABC
import numpy as np
from collections import namedtuple
from sort.sort import Sort


from detector import DetectorOutput


TrackerOutput = namedtuple('TrackerOutput',
                           ['bbox_xywh', 'ids'])


class Tracker(ABC):
    """
    Abstract class tracker. Its method `update` takes detector outputs and returns tracked objects
    """
    def __init__(self, cli_args: argparse.Namespace or object, *args, **kwargs):
        pass

    def update(self, img: np.array, detector_outputs: DetectorOutput) -> TrackerOutput:
        """
        Call tracker's updating and get output in in form of named tuple "TrackerOutput".
        It consists of 2 fields (N is number of tracks):

        'bbox_xywh':    np.array of bounding boxes with shape (N, 4),
                        which has format xywh (i.e. top-left corner and width and height);
        'ids':          class id's, np.array with shape (N, 1);
        """
        pass


class TrackerSort(Tracker):
    def __init__(self, argv, *args, **kwargs):
        super(TrackerSort, self).__init__(argv, *args, **kwargs)
        self.tracker = Sort(max_age=25)

    def update(self, img: np.ndarray, detector_outputs: DetectorOutput) -> TrackerOutput:
        tracker_input = np.concatenate((np.reshape(xywh_to_xyxy(detector_outputs.bbox_xywh), newshape=(-1, 4)),
                                        np.reshape(detector_outputs.cls_conf, newshape=(-1, 1))),
                                       axis=1)
        tracker_output = self.tracker.update(tracker_input)
        bbox_xyxy = tracker_output[:, :4]
        bbox_xywh = xyxy_to_xywh(bbox_xyxy)
        identities = tracker_output[:, 4]
        return TrackerOutput(bbox_xywh, identities)


def xyxy_to_xywh(xyxy: np.ndarray):
    xywh = np.zeros(xyxy.shape)
    for i in range(xyxy.shape[0]):
        xywh[i][0] = xyxy[i][0]
        xywh[i][1] = xyxy[i][1]
        xywh[i][2] = xyxy[i][2] - xyxy[i][0]
        xywh[i][3] = xyxy[i][3] - xyxy[i][1]
    return xywh


def xywh_to_xyxy(xywh: np.ndarray):
    xyxy = np.zeros(xywh.shape)
    for i in range(xywh.shape[0]):
        xyxy[i][0] = xywh[i][0]
        xyxy[i][1] = xywh[i][1]
        xyxy[i][2] = xywh[i][0] + xywh[i][2]
        xyxy[i][3] = xywh[i][1] + xywh[i][3]
    return xyxy


def xcycwh_to_xywh(xcycwh: np.ndarray):
    xywh = np.zeros(xcycwh.shape)
    for i in range(xcycwh.shape[0]):
        xywh[i][0] = xcycwh[i][0] - xcycwh[i][2] / 2
        xywh[i][1] = xcycwh[i][1] - xcycwh[i][3] / 2
        xywh[i][2] = xcycwh[i][2]
        xywh[i][3] = xcycwh[i][3]
    return xywh


def xywh_to_xcycwh(xywh: np.ndarray):
    xcycwh = np.zeros(xywh.shape)
    for i in range(xywh.shape[0]):
        xcycwh[i][0] = xywh[i][0] + xywh[i][2] / 2
        xcycwh[i][1] = xywh[i][1] + xywh[i][3] / 2
        xcycwh[i][2] = xywh[i][2]
        xcycwh[i][3] = xywh[i][3]
    return xcycwh


