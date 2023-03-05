import argparse

import cv2
import time
import numpy as np
import logging
import os
import math

from detector import Detector, DetectorOutput
from tracker import Tracker
from face_recognitor import FaceRecognitor
from drawer import Drawer
from logger import get_logger
from skvideo.io import FFmpegWriter

project_dir = os.path.join(os.path.dirname(__file__))


def get_absolute_path(project_dir: str, path: str):

    # if path is absolute, then just return it
    if path[0] == '/' or (path[0].isupper() and path[1:3] == ':\\'):
        return path

    # if path is relative, merge it with project_dir
    abs_path = os.path.join(project_dir, path)
    return abs_path


class Timer(object):
    """A simple timer"""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        return self.average_time, self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.


class Tracking:
    """
    Class of tracking. Instance of this class implements tracking of object in one sequence (video, img)
    """
    def __init__(self, args: argparse.Namespace, detector: Detector, tracker: Tracker):
        self.args = args
        self.detector = detector
        self.tracker = tracker
        
        self.face_recognitor = FaceRecognitor('./faces/')
        self.id_face_conformity = {}
        self.timer = Timer()
        self.drawer = Drawer()

        self.logger = get_logger(__name__, logging.INFO)
        self.path = self.args.path #get_absolute_path(project_dir, self.args.path)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        elif args.data_type == "video":
            self.vdo = cv2.VideoCapture()
        else:
            pass

    def __enter__(self):

        if self.args.path is None and self.args.cam == -1 and self.args.detector == 'gt':

            self.number_of_frames = self.detector.gt_data_frame.values[:, 0].max()
            self.im_width = 0
            self.im_height = 0

        elif self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        elif self.args.data_type == "video":
            assert os.path.isfile(self.path), "Path error"
            self.vdo.open(self.path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        elif self.args.data_type == "images":
            assert os.path.isdir(self.path), "Path error"
            self.filenames = os.listdir(self.path)
            self.filenames.sort(key=lambda x: int(x.split('.')[0]))
            first_image = cv2.imread(os.path.join(self.path, self.filenames[0]))
            self.im_width = first_image.shape[1]
            self.im_height = first_image.shape[0]
            self.image_cnt = 0
        else:
            raise ValueError("Wrong data_type")

        if self.args.save_video_path:
            self.args.save_video_path = get_absolute_path(project_dir, self.args.save_video_path)
            os.makedirs(os.path.join(*os.path.split(self.args.save_video_path)[:-1]), exist_ok=True)
            # path of saved video
            self.save_video_path = self.args.save_video_path
            # create video writer
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            # self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))
            self.writer = FFmpegWriter(self.save_video_path)
            # logging
            self.logger.info("Save video results to {}".format(self.args.save_video_path))

        if self.args.save_mot_path:
            self.args.save_mot_path = get_absolute_path(project_dir, self.args.save_mot_path)
            os.makedirs(os.path.join(*os.path.split(self.args.save_mot_path)[:-1]), exist_ok=True)
            # path of saved video and results
            self.save_mot_path = self.args.save_mot_path
            self.logger.info("Save text results to {}".format(self.args.save_mot_path))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        # save results in MOT format
        if self.args.save_mot_path:
            self.write_results(self.save_mot_path, self.results, 'mot')

        # self.logger.handlers = []
        self.logger.manager.loggerDict.pop(__name__)
        self.writer.close()
        cv2.destroyAllWindows()
        if exc_type:
            print(exc_type, exc_val, exc_tb)

    def run(self):
        self.results = []
        self.idx_frame = 0
        pressed_key = -1

        self.logger.info("Tracking is starting")
        while self._tracking_is_not_stopped():
            self.idx_frame += 1

            orig_img = self._get_orig_img()
            img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)     # maybe unnecessary

            # do detection
            detector_outputs = self.detector(img, frame_id=self.idx_frame)
            
            # filter dets
            bbox_xywh, cls_conf, cls_ids = detector_outputs
            new_bbox_xywh, new_cls_conf, new_cls_ids = [], [], []
            for i in range(len(bbox_xywh)):
                if cls_ids[i] == 0:
                    new_bbox_xywh.append(bbox_xywh[i])
                    new_cls_conf.append(cls_conf[i])
                    new_cls_ids.append(cls_ids[i])
            detector_outputs = DetectorOutput(np.array(new_bbox_xywh), 
                                              np.array(new_cls_conf), 
                                              np.array(new_cls_ids))

            # do tracking
            self.timer.tic()
            tracker_outputs = self.tracker.update(img, detector_outputs)
            average, diff = self.timer.toc()

            bbox_xywh, identities = tracker_outputs
            self.results.append((self.idx_frame, bbox_xywh, identities))

            # face recognition
            for i, id in enumerate(identities):
                
                # If face's already been in dict then skip
                if id in self.id_face_conformity:
                    continue
                
                # Find all faces in object crop
                x, y, w, h = map(int, bbox_xywh[i])
                obj_img = img[y: y + h, x: x + w]
                face_locations_xywh, face_names = self.face_recognitor(obj_img)
                
                face_number = -1
                max_sq = 0
                for i in range(len(face_names)):
                    if face_names[i] == 'Unknown':
                        continue
                    x, y, w, h = face_locations_xywh[i]
                    sq = w * h
                    if sq > max_sq:
                        max_sq = sq
                        face_number = i
                
                # If there is no faces, then skip it
                if face_number == -1:
                    continue
                
                # Take max square face as face of person in the object 
                current_name = face_names[face_number]
                self.id_face_conformity[id] = current_name
                
            
            # summarize info about the current frame
            info = {
                'frame index': str(self.idx_frame),
                # 'detector': self.args.detector,
                # 'tracker': self.args.tracker,
                'time': '{:.06f} s'.format(average),
                # 'fps': '{:.03f}'.format((1 / average) if average != 0 else math.inf),
                'detection numbers': str(detector_outputs.bbox_xywh.shape[0]),
                'tracking numbers': str(tracker_outputs.bbox_xywh.shape[0]),
            }

            # draw bboxes and info
            orig_img = self.drawer(orig_img, info, tracker_outputs, self.id_face_conformity)

            # logging
            if self.idx_frame % 100 == 0:
                self.logger.info(', '.join([f'{key}: {info[key]}' for key in info.keys()]))
            else:
                self.logger.debug(', '.join([f'{key}: {info[key]}' for key in info.keys()]))

            # display in window
            if self.args.display:
                cv2.imshow("test", orig_img)
                pressed_key = cv2.waitKey(1)

            # save video frame
            if self.args.save_video_path:
                self.writer.writeFrame(orig_img[:, :, ::-1])

            if pressed_key == 27:
                break

        self.logger.info("Tracking is finished")

    def _tracking_is_not_stopped(self) -> bool:
        if self.args.path is None and self.args.cam == -1 and self.args.detector == 'gt':
            return self.idx_frame < self.number_of_frames

        if self.args.data_type == "video":
            return self.vdo.grab()

        elif self.args.data_type == "images":
            return self.image_cnt < len(self.filenames)

    def _get_orig_img(self) -> np.ndarray or None:
        orig_img = None
        if self.args.path is None and self.args.cam == -1 and self.args.detector == 'gt':
            orig_img = np.zeros((1, 1, 3), dtype=np.float32)

        elif self.args.data_type == "video":
            _, orig_img = self.vdo.retrieve()

        elif self.args.data_type == "images":
            image_path = os.path.join(self.path, self.filenames[self.image_cnt])
            self.image_cnt += 1
            orig_img = cv2.imread(image_path)
        return orig_img

    @staticmethod
    def write_results(filename, results, data_type):
        if data_type == 'mot':
            save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in sorted(zip(tlwhs, track_ids), key=lambda x: x[1]):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
