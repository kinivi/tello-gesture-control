from djitellopy import Tello

class TelloGestureController:
    def __init__(self, tello: Tello):
        self.tello = tello

    def gesture_control(self, gesture_buffer):
        gesture_id = gesture_buffer.get_gesture()
        print("GESTURE", gesture_id)
        if gesture_id == 0:  # ESC
            self.tello.move_forward(20)
        elif gesture_id == 1:
            self.tello.move_back(20)
        elif gesture_id == 2:
            self.tello.move_up(15)
        elif gesture_id == 3:
            self.tello.land()
        # elif key == ord('s'):
        #     tello.move_back(30)
        # elif key == ord('a'):
        #     tello.move_left(30)
        # elif key == ord('d'):
        #     tello.move_right(30)
        # elif key == ord('e'):
        #     tello.rotate_clockwise(30)
        # elif key == ord('q'):
        #     tello.rotate_counter_clockwise(30)
        # elif key == ord('r'):
        #     tello.move_up(30)
        # elif key == ord('f'):
        #     tello.move_down(30)



