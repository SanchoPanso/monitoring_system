import cv2
import numpy as np

from tracker import TrackerOutput


class Drawer:
    def __init__(self):
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    def __call__(self,
                 img: np.ndarray,
                 info: dict,
                 tracker_output: TrackerOutput,
                 id_face_conformity: dict,
                 class_names_list: list = None):

        # draw boxes for visualization
        if len(tracker_output.ids) > 0:
            img = self.draw_boxes(img, tracker_output, id_face_conformity)

        # draw info
        for i, key in enumerate(info.keys()):
            row = f'{key}: {info[key]}'
            t_size = cv2.getTextSize(row, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)[0]
            cv2.putText(img, row, (0, int(t_size[1] * (2 * i + 1.5))),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=2)
        return img

    def draw_boxes(self, img, tracker_output: TrackerOutput, id_face_conformity, class_names_list: list = None, offset=(0, 0)):
        bbox_xywh, identities = tracker_output
        for i, box in enumerate(bbox_xywh):
            x, y, w, h = [int(i) for i in box]
            x += offset[0]
            y += offset[1]

            # box text and bar
            id = int(identities[i])
            color = self.compute_color_for_labels(id)

            name = 'Unknown' if id not in id_face_conformity else id_face_conformity[id]
            label = f"{str(id)} {name}"

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.rectangle(img, (x, y), (x + t_size[0] + 3, y + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x, y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        return img

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)
