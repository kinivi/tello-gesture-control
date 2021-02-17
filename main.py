#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configargparse

import cv2 as cv

from gestures.tello_gesture_controller import TelloGestureController
from utils import CvFpsCalc

from djitellopy import Tello
from gestures import *

import threading


def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add("--is_keyboard", help='To use Keyboard control by default', type=bool)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)

    args = parser.parse_args()

    return args


def main():
    # init global vars
    global gesture_buffer
    global gesture_id

    # Argument parsing
    args = get_args()
    KEYBOARD_CONTROL = args.is_keyboard

    # Camera preparation
    tello = Tello()
    tello.connect()
    tello.streamon()

    print(tello.get_battery())

    # Take-off drone
    tello.takeoff()

    cap = tello.get_frame_read()

    # Init Tello Controllers
    gesture_controller = TelloGestureController(tello)
    keyboard_controller = TelloKeyboardController(tello)

    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=5)

    def tello_control(key, keyboard_controller, gesture_controller):
        global gesture_buffer

        if KEYBOARD_CONTROL:
            keyboard_controller.control(key)
        else:
            gesture_controller.gesture_control(gesture_buffer)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1

    tello.move_down(20)

    while True:
        fps = cv_fps_calc.get()
        # tello.send_command_without_return("go 0 0 0 0")

        # Process Key (ESC: end)
        key = cv.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        elif key == ord('k'):
            KEYBOARD_CONTROL = True
        elif key == ord('g'):
            KEYBOARD_CONTROL = False

        # Camera capture
        image = cap.frame

        debug_image, gesture_id = gesture_detector.recognize(image)
        gesture_buffer.add_gesture(gesture_id)

        #Start control thread
        threading.Thread(target=tello_control, args=(key, keyboard_controller, gesture_controller, )).start()

        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)

        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)

    tello.land()
    tello.end()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
