import cv2
import numpy as np
import pytesseract
import re
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import subprocess
import os
from pathlib import Path
import json
from config import ConfigManager


@dataclass
class ChampionDetectionResult:
    champion_name: Optional[str]
    confidence: float
    processed_region: Optional[np.ndarray]
    raw_text: str
    success: bool


class ChampionNameExtractor:
    def __init__(self, config_path: str = "config.json"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_champion_detection_config()

    def extract_champion_region(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        region_config = self.config["region"]

        bottom_height = int(height * region_config["bottom_offset_ratio"])
        top_y = height - bottom_height

        horizontal_width = int(width * region_config["horizontal_center_ratio"])
        left_x = (width - horizontal_width) // 2
        right_x = left_x + horizontal_width

        vertical_height = int(bottom_height * region_config["vertical_center_ratio"])
        bottom_y = height
        adjusted_top_y = bottom_y - vertical_height

        roi = image[adjusted_top_y:bottom_y, left_x:right_x]

        if "top_trim_pixels" in region_config:
            trim_pixels = region_config["top_trim_pixels"]
            if trim_pixels > 0 and trim_pixels < roi.shape[0]:
                roi = roi[trim_pixels:, :]

        if "bottom_trim_ratio" in region_config:
            trim_ratio = region_config["bottom_trim_ratio"]
            if 0 < trim_ratio < 1:
                roi_height = roi.shape[0]
                trim_height = int(roi_height * (1 - trim_ratio))
                roi = roi[:trim_height, :]

        return roi

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        preprocessing_config = self.config["preprocessing"]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        processed = gray.copy()

        if preprocessing_config.get("invert_for_white_text", True):
            mean_intensity = np.mean(processed)
            if mean_intensity < 128:
                processed = cv2.bitwise_not(processed)

        if preprocessing_config.get("enhance_contrast", True):
            alpha = preprocessing_config["contrast_alpha"]
            enhanced = cv2.convertScaleAbs(processed, alpha=alpha, beta=0)
        else:
            enhanced = processed

        if preprocessing_config.get("use_adaptive_threshold", True):
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            threshold_val = preprocessing_config["threshold_value"]
            _, thresh = cv2.threshold(enhanced, threshold_val, 255, cv2.THRESH_BINARY)

        return thresh

    def extract_text_from_region(self, processed_image: np.ndarray) -> Tuple[str, float]:
        try:
            scale_factor = self.config["ocr"].get("scale_factor", 3)
            if scale_factor > 1:
                height, width = processed_image.shape
                new_height, new_width = height * scale_factor, width * scale_factor
                scaled_image = cv2.resize(processed_image, (new_width, new_height),
                                          interpolation=cv2.INTER_CUBIC)
            else:
                scaled_image = processed_image

            best_config = "--psm 6"

            try:
                data = pytesseract.image_to_data(
                    scaled_image,
                    config=best_config,
                    output_type=pytesseract.Output.DICT
                )

                words = []
                confidences = []

                for i in range(len(data['text'])):
                    word = data['text'][i].strip()
                    conf = float(data['conf'][i])

                    if word and conf > self.config["ocr"]["min_confidence"]:
                        words.append(word)
                        confidences.append(conf)

                extracted_text = " ".join(words)
                avg_confidence = np.mean(confidences) if confidences else 0.0

                if extracted_text and avg_confidence > self.config["ocr"]["min_confidence"]:
                    return extracted_text, avg_confidence

            except Exception as e:
                pass

            try:
                simple_text = pytesseract.image_to_string(scaled_image, config="--psm 6").strip()
                if simple_text:
                    return simple_text, 70.0
            except Exception as e:
                pass

            return "", 0.0

        except Exception as e:
            return "", 0.0

    def clean_champion_name(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        text_config = self.config["text_processing"]

        cleaned = re.sub(r'[^\w\s\-]', '', raw_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            return None

        words = cleaned.split()
        suffixes = text_config["common_skin_suffixes"]
        filtered_words = []

        for word in words:
            if (len(word) >= text_config["min_word_length"] and
                    word.lower() not in suffixes and
                    word.upper() not in ["RANDOM", "SKIN"]):
                filtered_words.append(word)

        if not filtered_words:
            return None

        champion_name = filtered_words[-1]
        champion_name = champion_name.capitalize()

        return champion_name

    def detect_champion_from_screenshot(self, screenshot_path: str) -> ChampionDetectionResult:
        try:
            image = cv2.imread(screenshot_path)
            if image is None:
                return ChampionDetectionResult(
                    champion_name=None,
                    confidence=0.0,
                    processed_region=None,
                    raw_text="",
                    success=False
                )

            return self.detect_champion_from_image(image)

        except Exception as e:
            return ChampionDetectionResult(
                champion_name=None,
                confidence=0.0,
                processed_region=None,
                raw_text="",
                success=False
            )

    def detect_champion_from_image(self, image: np.ndarray) -> ChampionDetectionResult:
        try:
            roi = self.extract_champion_region(image)
            processed = self.preprocess_image(roi)

            scale_factor = self.config["ocr"].get("scale_factor", 3)
            if scale_factor > 1:
                height, width = processed.shape
                new_height, new_width = height * scale_factor, width * scale_factor
                scaled_for_debug = cv2.resize(processed, (new_width, new_height),
                                              interpolation=cv2.INTER_CUBIC)
            else:
                scaled_for_debug = processed

            raw_text, confidence = self.extract_text_from_region(processed)
            champion_name = self.clean_champion_name(raw_text)
            success = champion_name is not None and confidence > self.config["ocr"]["min_confidence"]

            result = ChampionDetectionResult(
                champion_name=champion_name,
                confidence=confidence,
                processed_region=processed,
                raw_text=raw_text,
                success=success
            )

            self.save_debug_images(result, "debug", roi, scaled_for_debug)
            self.save_preprocessing_steps(roi)

            return result

        except Exception as e:
            return ChampionDetectionResult(
                champion_name=None,
                confidence=0.0,
                processed_region=None,
                raw_text="",
                success=False
            )

    def save_debug_images(self, result: ChampionDetectionResult, output_prefix: str = "debug",
                          original_region: Optional[np.ndarray] = None,
                          scaled_image: Optional[np.ndarray] = None) -> None:
        try:
            if original_region is not None:
                cv2.imwrite(f"{output_prefix}_original.png", original_region)

            if scaled_image is not None:
                cv2.imwrite(f"{output_prefix}_scaled_for_ocr.png", scaled_image)

        except Exception as e:
            pass

    def save_preprocessing_steps(self, image: np.ndarray, output_prefix: str = "debug_steps") -> None:
        try:
            preprocessing_config = self.config["preprocessing"]

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            cv2.imwrite(f"{output_prefix}_1_grayscale.png", gray)

            processed = gray.copy()

            if preprocessing_config.get("invert_for_white_text", True):
                mean_intensity = np.mean(processed)
                if mean_intensity < 128:
                    inverted = cv2.bitwise_not(processed)
                    cv2.imwrite(f"{output_prefix}_2_inverted.png", inverted)
                    processed = inverted
                else:
                    cv2.imwrite(f"{output_prefix}_2_no_invert_needed.png", processed)

            if preprocessing_config.get("enhance_contrast", True):
                alpha = preprocessing_config["contrast_alpha"]
                enhanced = cv2.convertScaleAbs(processed, alpha=alpha, beta=0)
            else:
                enhanced = processed
            cv2.imwrite(f"{output_prefix}_3_contrast_only.png", enhanced)

            if preprocessing_config.get("use_adaptive_threshold", True):
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                threshold_val = preprocessing_config["threshold_value"]
                _, thresh = cv2.threshold(enhanced, threshold_val, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f"{output_prefix}_4_final_threshold.png", thresh)

        except Exception as e:
            pass


class ChampionDetector:
    def __init__(self, config_path: str = "config.json"):
        self.extractor = ChampionNameExtractor(config_path)

    def get_champion_from_file(self, screenshot_path: str) -> Optional[str]:
        result = self.extractor.detect_champion_from_screenshot(screenshot_path)
        return result.champion_name if result.success else None

if __name__ == "__main__":
    extractor = ChampionDetector()
    result = extractor.get_champion_from_file("champ_select.webp")
    print(result)