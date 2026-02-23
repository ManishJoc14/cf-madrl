import cv2
import numpy as np


class DisplayManager:
    """Combine multiple lane frames into one window for clean display"""

    def __init__(self, window_name="Traffic Monitor"):
        self.window_name = window_name
        self.frames = {}
        self.running = True

    def update_frame(self, lane_id, frame):
        self.frames[lane_id] = frame

    def show_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while self.running:
            if self.frames:
                # Stack frames vertically
                ordered_frames = [self.frames[k] for k in sorted(self.frames.keys())]
                resized = [cv2.resize(f, (640, 480)) for f in ordered_frames]
                combined = np.vstack(resized)
                cv2.imshow(self.window_name, combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        cv2.destroyAllWindows()
