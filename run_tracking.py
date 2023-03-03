import os
import sys

from arg_parser import TrackerArgParser
from tracking import Tracking
from detector import DetectorYolov5
from tracker import TrackerSort


if __name__ == '__main__':

    # parse arguments
    parser = TrackerArgParser()
    args = parser.parse_args()
    
    args.display = True

    # create instances of detector and tracker
    detector = DetectorYolov5(args)
    tracker = TrackerSort(args)

    # run tracking
    with Tracking(args, detector, tracker) as trk:
        trk.run()

