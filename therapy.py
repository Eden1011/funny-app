import sys
import os
import dlib
import numpy as np
import cv2
import json
import time
from chill import chill
from config import *


class FaceRegionExtractor:
    def __init__(self, config):
        self.config = config

    def extract_region(self, frame, landmarks, region_name):
        try:
            landmark_config = self.config["landmarks"][region_name]
            padding = self.config["regions"][region_name]["padding"]

            region_points = []
            for i in range(landmark_config["start"], landmark_config["end"] + 1):
                point = landmarks.part(i)
                region_points.append([point.x, point.y])

            region_points = np.array(region_points)

            x_min = np.min(region_points[:, 0]) - padding
            x_max = np.max(region_points[:, 0]) + padding
            y_min = np.min(region_points[:, 1]) - padding
            y_max = np.max(region_points[:, 1]) + padding

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            return region, (int(x_min), int(y_min), int(x_max), int(y_max))

        except Exception as e:
            print(f"Error extracting region {region_name}: {e}")
            return None, None

    def draw_region_box(self, frame, box, region_name):
        if box is not None:
            color = self.config["colors"][f"{region_name}_box"]
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, region_name.replace('_', ' ').title(), (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class SmileDetector:
    def __init__(self, config):
        self.config = config

    def calculate_eye_aspect_ratio(self, landmarks, eye_type):
        landmark_config = self.config["landmarks"][eye_type]
        eye_points = np.array([[landmarks.part(i).x, landmarks.part(i).y]
                               for i in range(landmark_config["start"], landmark_config["end"])])

        eye_width = np.linalg.norm(eye_points[3] - eye_points[0])
        eye_height = np.linalg.norm(eye_points[1] - eye_points[4])

        return eye_width / eye_height

    def calculate_mouth_aspect_ratio(self, landmarks):
        landmark_config = self.config["landmarks"]["mouth"]
        mouth_points = np.array([[landmarks.part(i).x, landmarks.part(i).y]
                                 for i in range(landmark_config["start"], landmark_config["end"] + 1)])

        mouth_width = np.linalg.norm(mouth_points[6] - mouth_points[0])
        mouth_height = np.linalg.norm(mouth_points[9] - mouth_points[3])

        divisor = self.config["smile_detection"]["mouth_ratio_divisor"]
        return mouth_width / mouth_height / divisor

    def detect_smile(self, landmarks):
        left_eye_ratio = self.calculate_eye_aspect_ratio(landmarks, "left_eye")
        right_eye_ratio = self.calculate_eye_aspect_ratio(landmarks, "right_eye")
        mouth_ratio = self.calculate_mouth_aspect_ratio(landmarks)

        mouth_threshold = self.config["smile_detection"]["mouth_ratio_threshold"]
        eyes_threshold = self.config["smile_detection"]["eyes_ratio_threshold"]

        mouth_condition = mouth_ratio > mouth_threshold
        eyes_ratio_mean = (left_eye_ratio + right_eye_ratio) / 2
        eyes_condition = eyes_ratio_mean > eyes_threshold

        return mouth_condition and eyes_condition, mouth_ratio, eyes_ratio_mean


class HappinessMeter:
    def __init__(self, config):
        self.config = config
        self.width, self.height = config["display"]["meter_dimensions"]

    def create_meter(self):
        meter = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        cv2.rectangle(meter, (5, 5), (self.width - 5, self.height - 5), (0, 0, 0), 2)
        return meter

    def update_meter(self, meter, happiness_progress, zero_timer_info=None):
        meter_copy = meter.copy()

        meter_left = 40
        meter_right = self.width - 40
        meter_top = self.height // 2 - 15
        meter_bottom = self.height // 2 + 15
        meter_width = meter_right - meter_left

        cv2.rectangle(meter_copy, (meter_left, meter_top), (meter_right, meter_bottom), (200, 200, 200), -1)
        cv2.rectangle(meter_copy, (meter_left, meter_top), (meter_right, meter_bottom), (0, 0, 0), 2)

        fill_width = int(meter_width * happiness_progress)

        if fill_width > 0:
            if happiness_progress <= 0.5:
                color_ratio = happiness_progress * 2
                color = (0, int(255 * color_ratio), int(255 * (1 - color_ratio)))
            else:
                color_ratio = (happiness_progress - 0.5) * 2
                color = (0, 255, int(255 * (1 - color_ratio)))

            cv2.rectangle(meter_copy, (meter_left, meter_top), (meter_left + fill_width, meter_bottom), color, -1)

        cv2.putText(meter_copy, "0", (10, self.height // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(meter_copy, "100", (self.width - 35, self.height // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                    2)

        current_value = int(happiness_progress * 100)
        value_text = str(current_value)
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (self.width - text_size[0]) // 2
        cv2.putText(meter_copy, value_text, (text_x, self.height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0),
                    2)

        if zero_timer_info and happiness_progress == 0.0:
            time_at_zero, triggered = zero_timer_info
            if not triggered:
                remaining_time = self.config["happiness"]["zero_action_delay"] - time_at_zero
                if remaining_time > 0:
                    countdown_text = f"Action in: {remaining_time:.1f}s"
                    cv2.putText(meter_copy, countdown_text, (self.width // 2 - 80, self.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(meter_copy, "ACTION TRIGGERED!", (self.width // 2 - 90, self.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return meter_copy


class SmileDisplay:
    def __init__(self, config):
        self.config = config
        self.width, self.height = config["display"]["smile_display_dimensions"]

    def create_display(self):
        display = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        cv2.rectangle(display, (10, 10), (self.width - 10, self.height - 10), (0, 0, 0), 2)
        return display

    def update_display(self, display, is_smiling, mouth_ratio, eye_ratio):
        display_copy = display.copy()

        title = "SMILE DETECTOR"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        title_x = (self.width - title_size[0]) // 2
        cv2.putText(display_copy, title, (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if is_smiling:
            status = "SMILING! :D"
            status_color = (0, 255, 0)
        else:
            status = "NOT SMILING :("
            status_color = (0, 0, 255)

        status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        status_x = (self.width - status_size[0]) // 2
        cv2.putText(display_copy, status, (status_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        y_pos = 120
        cv2.putText(display_copy, f"Mouth Ratio: {mouth_ratio:.2f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        cv2.putText(display_copy, f"Threshold: {self.config['smile_detection']['mouth_ratio_threshold']:.2f}",
                    (220, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        y_pos += 25
        cv2.putText(display_copy, f"Eye Ratio: {eye_ratio:.2f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        cv2.putText(display_copy, f"Threshold: {self.config['smile_detection']['eyes_ratio_threshold']:.2f}",
                    (220, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        y_pos += 35
        mouth_status = "YES" if mouth_ratio > self.config['smile_detection']['mouth_ratio_threshold'] else "NO"
        eye_status = "YES" if eye_ratio > self.config['smile_detection']['eyes_ratio_threshold'] else "NO"

        cv2.putText(display_copy, f"Mouth: {mouth_status}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0) if mouth_status == "YES" else (0, 0, 150), 2)
        cv2.putText(display_copy, f"Eyes: {eye_status}", (150, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0) if eye_status == "YES" else (0, 0, 150), 2)

        return display_copy


class Therapist:
    def __init__(self, config_path="config.json"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.config["predictor_path"])

        self.region_extractor = FaceRegionExtractor(self.config)
        self.smile_detector = SmileDetector(self.config)
        self.happiness_meter = HappinessMeter(self.config)
        self.smile_display = SmileDisplay(self.config)

        self.current_happiness = self.config["happiness"]["start_value"]
        self.last_update_time = time.time()
        self.zero_happiness_start_time = None
        self.zero_happiness_triggered = False

    def setup_windows(self):
        windows = ['Main Feed', 'Left Eye', 'Right Eye', 'Mouth', 'Smile Detection', 'Happiness Meter']

        for window in windows:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

        positions = self.config["windows"]["positions"]
        sizes = self.config["windows"]["sizes"]

        cv2.moveWindow('Main Feed', *positions["main_feed"])
        cv2.moveWindow('Left Eye', *positions["left_eye"])
        cv2.moveWindow('Right Eye', *positions["right_eye"])
        cv2.moveWindow('Mouth', *positions["mouth"])
        cv2.moveWindow('Smile Detection', *positions["smile_detection"])
        cv2.moveWindow('Happiness Meter', *positions["happiness_meter"])

        cv2.resizeWindow('Left Eye', *sizes["left_eye"])
        cv2.resizeWindow('Right Eye', *sizes["right_eye"])
        cv2.resizeWindow('Mouth', *sizes["mouth"])
        cv2.resizeWindow('Smile Detection', *sizes["smile_detection"])
        cv2.resizeWindow('Happiness Meter', *sizes["happiness_meter"])

    def zero_happiness_action(self):
        chill()
        exit(0)

    def update_happiness(self, is_smiling):
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        self.last_update_time = current_time

        if is_smiling:
            self.current_happiness = min(100.0, self.current_happiness +
                                         self.config["happiness"]["fill_rate"] * time_diff * 100)
        else:
            self.current_happiness = max(0.0, self.current_happiness -
                                         self.config["happiness"]["decay_rate"] * time_diff * 100)

        if self.current_happiness == 0.0:
            if self.zero_happiness_start_time is None:
                self.zero_happiness_start_time = time.time()
                self.zero_happiness_triggered = False
                print("Happiness reached 0! Starting timer...")

            time_at_zero = time.time() - self.zero_happiness_start_time
            if time_at_zero >= self.config["happiness"]["zero_action_delay"] and not self.zero_happiness_triggered:
                self.zero_happiness_action()
                self.zero_happiness_triggered = True
        else:
            if self.zero_happiness_start_time is not None:
                print(f"Happiness recovered! Was at 0 for {time.time() - self.zero_happiness_start_time:.1f} seconds")
                self.zero_happiness_start_time = None
                self.zero_happiness_triggered = False

    def run(self):
        cap = cv2.VideoCapture(0)
        self.setup_windows()

        meter_display = self.happiness_meter.create_meter()
        smile_info_display = self.smile_display.create_display()

        print("Modular Happiness Detector started!")
        print("Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if self.config["display"]["flip_horizontal"]:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            is_smiling = False

            for k, face in enumerate(faces):
                landmarks = self.predictor(gray, face)

                left_eye, left_eye_box = self.region_extractor.extract_region(frame, landmarks, "left_eye")
                right_eye, right_eye_box = self.region_extractor.extract_region(frame, landmarks, "right_eye")
                mouth, mouth_box = self.region_extractor.extract_region(frame, landmarks, "mouth")

                is_smiling, mouth_ratio, eye_ratio = self.smile_detector.detect_smile(landmarks)

                for num in range(landmarks.num_parts):
                    cv2.circle(frame, (landmarks.part(num).x, landmarks.part(num).y), 2,
                               self.config["colors"]["landmark_points"], -1)

                self.region_extractor.draw_region_box(frame, left_eye_box, "left_eye")
                self.region_extractor.draw_region_box(frame, right_eye_box, "right_eye")
                self.region_extractor.draw_region_box(frame, mouth_box, "mouth")

                face_color = (self.config["colors"]["face_smiling"] if is_smiling
                              else self.config["colors"]["face_not_smiling"])
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), face_color, 2)

                smile_text = "SMILING! :D" if is_smiling else "NOT SMILING :("
                cv2.putText(frame, smile_text, (face.left(), face.top() - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2)
                cv2.putText(frame, f"Face {k + 1}", (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

                scale_factor = self.config["display"]["scale_factor"]

                if left_eye is not None and left_eye.size > 0:
                    left_eye_big = cv2.resize(left_eye, None, fx=scale_factor, fy=scale_factor,
                                              interpolation=cv2.INTER_CUBIC)
                    cv2.imshow('Left Eye', left_eye_big)

                if right_eye is not None and right_eye.size > 0:
                    right_eye_big = cv2.resize(right_eye, None, fx=scale_factor, fy=scale_factor,
                                               interpolation=cv2.INTER_CUBIC)
                    cv2.imshow('Right Eye', right_eye_big)

                if mouth is not None and mouth.size > 0:
                    mouth_big = cv2.resize(mouth, None, fx=scale_factor, fy=scale_factor,
                                           interpolation=cv2.INTER_CUBIC)
                    cv2.imshow('Mouth', mouth_big)

                smile_info_updated = self.smile_display.update_display(smile_info_display, is_smiling,
                                                                       mouth_ratio, eye_ratio)
                cv2.imshow('Smile Detection', smile_info_updated)

            self.update_happiness(is_smiling)

            happiness_progress = self.current_happiness / 100.0
            zero_timer_info = None
            if self.zero_happiness_start_time is not None:
                time_at_zero = time.time() - self.zero_happiness_start_time
                zero_timer_info = (time_at_zero, self.zero_happiness_triggered)

            meter_updated = self.happiness_meter.update_meter(meter_display, happiness_progress, zero_timer_info)
            cv2.imshow('Happiness Meter', meter_updated)
            cv2.imshow('Main Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
