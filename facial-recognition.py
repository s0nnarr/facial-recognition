"""

Facial Recognition System using YOLO and OpenCV
This script captures video from the webcam, detects faces using YOLO & OpenCV
and recognizes known faces using face_recognition library.
It also allows for adding new known faces by placing images in the "known_faces" directory.

"""


import cv2
from ultralytics import YOLO
import os
import numpy as np
import face_recognition
from pathlib import Path


class FacialRecognition:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt')

        # Store known faces and names
        self.known_faces = []

        self.load_known_faces("known_faces/")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_known_faces(self, directory):
        """Load known faces from a directory."""

        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        try:
            faces_dir_path = Path(directory)
            if not os.path.exists(faces_dir_path):
                print("\n\n")
                print(f"Directory {faces_dir_path} does not exist. Creating it now.")
                print(f"The name of the files should be <name>.<extension> e.g. alex.jpg")
                print("\n\n")

                os.makedirs(faces_dir_path)
                return

            image_files = [f for f in faces_dir_path.iterdir()
                           if f.is_file() and f.name.endswith(tuple(allowed_extensions))]

            for image_file in image_files:
                try:
                    image = face_recognition.load_image_file(image_file)
                    face_encodings = face_recognition.face_encodings(image)
                    if len(face_encodings) > 0:
                        face_dict = {
                            "name": image_file.stem,
                            "encoding": face_encodings[0],
                            "file_path": str(image_file)
                        }

                        self.known_faces.append(face_dict)

                except Exception as err:
                    print(f"Error processing file {image_file}: {err}")
                    continue
        except Exception as e:
            print(f"Error loading known faces: {e}")

    def recognize_faces(self, frame, face_locations):
            """Recognize faces in the frame."""
            
            if not face_locations:
                return []
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                recognized_faces = []
                for face_encoding in face_encodings:
                    match = self.find_matching_face(face_encoding)
                    if match:
                        recognized_faces.append(match)
                return recognized_faces
            
            except Exception as err:
                print(f"Error during face recognition: {err}")
                return []

    def draw_results(self, frame, face_locations, recognized_faces):
        """Draw rectangles and labels around recognized faces."""
        try:
            if frame is None or not face_locations:
                print("No frame or face locations to draw.")
                return
            for i, (top, right, bottom, left) in enumerate(face_locations):
                top, right, bottom, left = int(top), int(right), int(bottom), int(left)
                if i < len(recognized_faces) and recognized_faces[i]:
                    face_info = recognized_faces[i]
                    label = f"{face_info['name']}"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

           
        except Exception as err:
            print(f"Error drawing results: {err}")

    def find_matching_face(self, face_encoding, tolerance=0.6):
        if not self.known_faces:
            return None
        
        known_encodings = [face['encoding'] for face in self.known_faces]
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        # argmin returns the index of the minimum value in the array
        # lower face distance means a better match
        
        if matches[best_match_index]:
            return self.known_faces[best_match_index]
        else:
            return None

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened:
            print("Cannot open camera")
            exit()

        print("-"*50)
        print("Press 'q' to quit the webcam window")
 
        flip_camera = True
        frame_count = 0
        
        last_face_locations = []
        last_recognized_faces = []
        current_face_locations = []
        current_recognized_faces = []
        detection_frame = 0


        def interpolate_positions(old_faces, new_faces, progress):
            """Interpolate positions of faces between old and new frames.
            This makes the transition smoother."""

            if not old_faces or not new_faces or len(old_faces) != len(new_faces):
                return new_faces
            interpolated = []
            for old_face, new_face in zip(old_faces, new_faces):
                old_t, old_r, old_b, old_l = old_face
                new_t, new_r, new_b, new_l = new_face

                # Linear interpolation
                interp_t = int(old_t + (new_t - old_t) * progress)
                interp_r = int(old_r + (new_r - old_r) * progress)
                interp_b = int(old_b + (new_b - old_b) * progress)
                interp_l = int(old_l + (new_l - old_l) * progress)

                interpolated.append((interp_t, interp_r, interp_b, interp_l))

            return interpolated

        while True:
            ret, frame = cap.read() # ret -> bool (If the frame was read successfully, return true else error);
                                    # frame = image data in the form of a matrix; e.g. frame[y][x] = [B,G,R]
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            if flip_camera:
                frame = cv2.flip(frame, 1)

                try:
                    if frame_count % 5 == 0:
                        # Detect faces in the frame
                        # Resize the frame to 1/4 to speed up processing
                        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        new_face_locations = face_recognition.face_locations(rgb_frame)
                        # print(f"Found {len(face_locations)}!")
                        # face_locations = [(top*4, right*4, bottom*4, left*4) # Resizing back to original
                        #                 for top, right, bottom, left in face_locations]
                        if new_face_locations:
                            # Store old positions for interpolation
                            last_face_locations = current_face_locations.copy()
                            current_face_locations = new_face_locations
                            detection_frame = frame_count

                            if self.known_faces:
                                current_recognized_faces = self.recognize_faces(frame, new_face_locations)
                            else:
                                current_recognized_faces = [None] * len(new_face_locations)
                        else:
                            # Gradually fade out the last known face locations
                            if frame_count - detection_frame > 15: 
                                current_face_locations = []
                                current_recognized_faces = []

                except Exception as err:
                    print(f"Error during face detection: {err}")
            if current_face_locations:
                frames_since_detection = frame_count - detection_frame
                if frames_since_detection == 0:
                    # Use exact locations, it was just detected.
                    smooth_face_locations = current_face_locations
                elif frames_since_detection < 5 and last_face_locations:
                    progress = frames_since_detection / 5.0
                    smooth_face_locations = interpolate_positions(last_face_locations, current_face_locations, progress)
                else:
                    smooth_face_locations = current_face_locations
                self.draw_results(frame, smooth_face_locations, current_recognized_faces)

            frame_count += 1                           

            cv2.imshow("Webcam", frame) # displays the frame in a window titled Webcam
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release() # release camera resource
        cv2.destroyAllWindows() # closes the opened Webcam window;

    
if __name__ == "__main__":

    fr = FacialRecognition()

    fr.load_known_faces("known_faces/")
    fr.run_detection();