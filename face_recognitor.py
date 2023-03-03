import face_recognition
import cv2
import os
import numpy as np
import glob


class FaceRecognitor:
    def __init__(self, face_dir: str):

        image_files = glob.glob(os.path.join(face_dir, '*.*'))
        self.known_face_encodings = []
        self.known_face_names = []
        
        for image_file in image_files:
            image = face_recognition.load_image_file(image_file)
            encoding = face_recognition.face_encodings(image)[0]
            name = os.path.splitext(os.path.split(image_file)[-1])[0]

            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
        
    
    def __call__(self, img: np.ndarray):
        rgb_img = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        face_names = []
        
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)
        
        face_locations_xywh = []
        for top, right, bottom, left in face_locations:
            x = left
            y = top
            w = right - left
            h = bottom - top
            face_locations_xywh.append([x, y, w, h])
        
        return face_locations_xywh, face_names
        


