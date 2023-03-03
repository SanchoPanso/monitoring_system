from argparse import ArgumentParser


class TrackerArgParser(ArgumentParser):
    def __init__(self):
        super(TrackerArgParser, self).__init__()

        self.add_argument("--detector", type=str, default='yolov5',
                          choices=['yolov3_opencv', 'yolov3', 'yolov5', 'gt'],
                          help='type of detector to be used')
        self.add_argument("--tracker", type=str, default='sort',
                          choices=['sort', 'deepsort', 'bytetrack', 'sdof', 'sort_modified', 'bytetrack_modified'],
                          help='type of tracker to be used')

        self.add_argument("--path", type=str, help='path to input video or directory with images')
        self.add_argument("--data_type", type=str, default="video", help='type of input (video or images)')

        self.add_argument("--display", action="store_true", help='create window with visualisation during processing')
        self.add_argument("--display_width", type=int, default=600, help='')
        self.add_argument("--display_height", type=int, default=400, help='')

        self.add_argument("--save_video_path", type=str, default="./output/result.avi",
                          help='path to saved output video')
        self.add_argument("--save_mot_path", type=str, default="./output/result.txt",
                          help='path to output text results in mot format')

        self.add_argument("--camera", action="store", dest="cam", type=int, default="1",
                          help='webcam id, value \'-1\' means not to use webcam')

        # self.add_argument("--config_detector", type=str, default=None,
        #                   help='path to cofiguration yaml-file for using detector. If None - use default')
        # self.add_argument("--config_tracker", type=str, default=None,
        #                   help='path to cofiguration yaml-file for using tracker. If None - use default')
        # self.add_argument("--gt_path", type=str, default=None,
        #                   help='path to ground-truth data for ground-truth detector (gt). '
        #                        'For other detectors it is ignored.'
        #                        'If None - use default')

