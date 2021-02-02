#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import cv2 as cv

from utils import CvFpsCalc

from djitellopy import Tello
from gesture_detector import GestureRecognition


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation
    tello = Tello()
    tello.connect()

    tello.streamon()
    cap = tello.get_frame_read()

    gesture_detector = GestureRecognition(use_static_image_mode, min_detection_confidence,
                                          min_tracking_confidence)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1

    while True:
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture
        image = cap.frame

        debug_image = gesture_detector.recognize(image)
        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)

        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
