import json


class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()

    def get_default_config(self):
        return {
            "therapy": {
                "landmarks": {
                    "left_eye": {"start": 36, "end": 41},
                    "right_eye": {"start": 42, "end": 47},
                    "mouth": {"start": 48, "end": 67}
                },
                "regions": {
                    "left_eye": {"padding": 15},
                    "right_eye": {"padding": 15},
                    "mouth": {"padding": 20}
                },
                "colors": {
                    "left_eye_box": [255, 0, 0],
                    "right_eye_box": [0, 255, 255],
                    "mouth_box": [0, 0, 255],
                    "face_smiling": [0, 255, 0],
                    "face_not_smiling": [0, 0, 255],
                    "landmark_points": [0, 255, 0]
                },
                "windows": {
                    "positions": {
                        "main_feed": [50, 50],
                        "left_eye": [700, 50],
                        "right_eye": [850, 50],
                        "mouth": [700, 200],
                        "smile_detection": [50, 400],
                        "happiness_meter": [50, 650]
                    },
                    "sizes": {
                        "left_eye": [300, 200],
                        "right_eye": [300, 200],
                        "mouth": [400, 300],
                        "smile_detection": [400, 200],
                        "happiness_meter": [500, 100]
                    }
                },
                "happiness": {
                    "decay_rate": 0.5,
                    "fill_rate": 0.5,
                    "start_value": 100.0,
                    "zero_action_delay": 5.0
                },
                "smile_detection": {
                    "mouth_ratio_threshold": 1.8,
                    "eyes_ratio_threshold": 2.3,
                    "mouth_ratio_divisor": 1.5
                },
                "display": {
                    "scale_factor": 5,
                    "flip_horizontal": True,
                    "meter_dimensions": [500, 100],
                    "smile_display_dimensions": [400, 200]
                },
                "predictor_path": "shape_predictor_81_face_landmarks.dat"
            },
            "champion_detection": {
                "region": {
                    "bottom_offset_ratio": 0.1,
                    "horizontal_center_ratio": 0.2,
                    "vertical_center_ratio": 4.0,
                    "top_trim_pixels": 50,
                    "bottom_trim_ratio": 0.7
                },
                "preprocessing": {
                    "contrast_alpha": 1.8,
                    "threshold_value": 180,
                    "invert_for_white_text": True,
                    "use_adaptive_threshold": True,
                    "enhance_contrast": True
                },
                "ocr": {
                    "config": "--psm 6",
                    "min_confidence": 20.0,
                    "scale_factor": 3
                },
                "text_processing": {
                    "min_word_length": 3,
                    "max_words": 4,
                    "common_skin_suffixes": ["skin", "chroma", "prestige", "edition", "random"]
                }
            }
        }

    def save_config(self, config_path=None):
        if config_path is None:
            config_path = self.config_path
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def create_default_config_file(self, config_path=None):
        if config_path is None:
            config_path = self.config_path
        default_config = self.get_default_config()
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)

    def get_therapy_config(self):
        return self.config.get("therapy", {})

    def get_champion_detection_config(self):
        return self.config.get("champion_detection", {})

    def update_therapy_config(self, updates):
        if "therapy" not in self.config:
            self.config["therapy"] = {}
        self.config["therapy"].update(updates)

    def update_champion_detection_config(self, updates):
        if "champion_detection" not in self.config:
            self.config["champion_detection"] = {}
        self.config["champion_detection"].update(updates)