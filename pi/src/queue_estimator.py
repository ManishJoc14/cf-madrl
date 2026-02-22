"""
Queue Length and Vehicle Waiting Time Estimator using Ultralytics YOLO

This module provides real-time estimation of queue length and average vehicle waiting times
using YOLO object detection and tracking. It integrates with Ultralytics solutions for
accurate traffic monitoring.
"""

import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO, solutions
from typing import Dict, List, Tuple, Optional


class VehicleTrack:
    """Represents a tracked vehicle with position history and waiting time calculation"""

    def __init__(self, track_id: int, initial_bbox: np.ndarray, fps: float = 30.0):
        self.track_id = track_id
        self.positions = deque(maxlen=30)  # Keep last 30 positions
        self.speeds = deque(maxlen=10)  # Keep last 10 speeds
        self.wait_start_time = None
        self.total_wait_time = 0.0
        self.is_waiting = False
        self.last_update = time.time()
        self.fps = fps

        # Initialize with first position
        self.update_position(initial_bbox)

    def update_position(self, bbox: np.ndarray):
        """Update vehicle position and calculate speed"""
        current_time = time.time()
        center = self._bbox_to_center(bbox)

        if self.positions:
            # Calculate speed based on position change
            prev_center = self.positions[-1]
            distance_pixels = np.linalg.norm(center - prev_center)
            time_diff = current_time - self.last_update

            # Convert to km/h (assuming meter_per_pixel calibration)
            # For now, use pixels per second, can be calibrated later
            speed_pixels_per_sec = distance_pixels / time_diff if time_diff > 0 else 0
            self.speeds.append(speed_pixels_per_sec)

        self.positions.append(center)
        self.last_update = current_time

        # Update waiting status
        self._update_waiting_status()

    def _bbox_to_center(self, bbox: np.ndarray) -> np.ndarray:
        """Convert bbox [x1,y1,x2,y2] to center point"""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _update_waiting_status(self):
        """Update waiting status based on speed"""
        if len(self.speeds) < 3:
            return

        # Average speed over last few frames
        avg_speed = np.mean(list(self.speeds))

        # Threshold for stopped vehicle (adjust based on calibration)
        stop_threshold = 5.0  # pixels per second

        current_time = time.time()

        if avg_speed < stop_threshold:
            if not self.is_waiting:
                self.wait_start_time = current_time
                self.is_waiting = True
        else:
            if self.is_waiting and self.wait_start_time:
                self.total_wait_time += current_time - self.wait_start_time
                self.is_waiting = False
                self.wait_start_time = None

    def get_current_wait_time(self) -> float:
        """Get current waiting time including ongoing wait"""
        wait_time = self.total_wait_time
        if self.is_waiting and self.wait_start_time:
            wait_time += time.time() - self.wait_start_time
        return wait_time

    def is_in_region(self, region: np.ndarray) -> bool:
        """Check if vehicle is in the specified region (polygon)"""
        if len(self.positions) == 0:
            return False

        center = self.positions[-1]
        # Convert to tuple of ints
        pt = (int(center[0]), int(center[1]))
        return cv2.pointPolygonTest(region.astype(np.float32), pt, False) >= 0


class QueueEstimator:
    """
    Estimates queue length and vehicle waiting times using YOLO tracking

    Args:
        model_path: Path to YOLO model
        regions: Dict of region polygons for each lane {lane_id: polygon}
        conf_threshold: Detection confidence threshold
        fps: Video FPS for time calculations
        meter_per_pixel: Calibration factor for distance
        stop_speed_threshold: Speed threshold for considering vehicle stopped (km/h)
        wait_time_threshold: Minimum time to consider vehicle waiting (seconds)
    """

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        regions: Optional[Dict[str, np.ndarray]] = None,
        conf_threshold: float = 0.3,
        fps: float = 30.0,
        meter_per_pixel: float = 0.05,
        stop_speed_threshold: float = 2.0,  # km/h
        wait_time_threshold: float = 3.0,  # seconds
    ):
        self.model = YOLO(model_path)
        self.regions = regions or {}
        self.conf_threshold = conf_threshold
        self.fps = fps
        self.meter_per_pixel = meter_per_pixel
        self.stop_speed_threshold = stop_speed_threshold
        self.wait_time_threshold = wait_time_threshold

        # Tracking state
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 0

        # Speed estimator for better speed calculation
        self.speed_estimator = solutions.SpeedEstimator(
            model=model_path,
            fps=fps,
            meter_per_pixel=meter_per_pixel,
        )

        # Vehicle classes (COCO: car, motorcycle, bus, truck)
        self.vehicle_classes = [1, 2, 3, 5, 7]

    def estimate(self, frame: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Process a frame and return queue estimates for each region

        Args:
            frame: Input video frame

        Returns:
            Dict with queue metrics per lane
        """
        # Run YOLO tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            classes=self.vehicle_classes,
            tracker="botsort.yaml"
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return self._get_empty_metrics()

        # Update tracks
        self._update_tracks(results[0].boxes, frame.shape[:2])

        # Calculate metrics per region
        metrics = {}
        for lane_id, region in self.regions.items():
            queue_length, avg_wait_time = self._calculate_lane_metrics(region)
            metrics[lane_id] = {
                "queue_length": queue_length,
                "avg_wait_time": avg_wait_time,
            }

        # Overall metrics
        total_queue = sum(m["queue_length"] for m in metrics.values())
        total_wait = sum(m["avg_wait_time"] * m["queue_length"] for m in metrics.values())
        avg_wait = total_wait / total_queue if total_queue > 0 else 0.0

        metrics["overall"] = {
            "queue_length": total_queue,
            "avg_wait_time": avg_wait,
        }

        return metrics

    def _update_tracks(self, boxes, frame_shape: Tuple[int, int]):
        """Update vehicle tracks with new detections"""
        current_track_ids = set()

        # Get boxes and ids separately like in monitor.py
        if boxes.id is not None:
            box_coords = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(box_coords, track_ids, clss):
                if cls not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = box

                current_track_ids.add(int(track_id))

                if int(track_id) not in self.tracks:
                    self.tracks[int(track_id)] = VehicleTrack(int(track_id), np.array([x1, y1, x2, y2]), self.fps)

                self.tracks[int(track_id)].update_position(np.array([x1, y1, x2, y2]))

        # Remove old tracks (not seen for a while)
        to_remove = []
        current_time = time.time()
        for track_id, track in self.tracks.items():
            if current_time - track.last_update > 5.0:  # 5 seconds timeout
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

    def _calculate_lane_metrics(self, region: np.ndarray) -> Tuple[int, float]:
        """Calculate queue length and average wait time for a lane region"""
        queue_vehicles = []
        total_wait = 0.0

        for track in self.tracks.values():
            if track.is_in_region(region):
                wait_time = track.get_current_wait_time()
                if wait_time >= self.wait_time_threshold:
                    queue_vehicles.append(track)
                    total_wait += wait_time

        queue_length = len(queue_vehicles)
        avg_wait = total_wait / queue_length if queue_length > 0 else 0.0

        return queue_length, avg_wait

    def _get_empty_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return empty metrics when no detections"""
        metrics = {}
        for lane_id in self.regions.keys():
            metrics[lane_id] = {"queue_length": 0, "avg_wait_time": 0.0}
        metrics["overall"] = {"queue_length": 0, "avg_wait_time": 0.0}
        return metrics

    def visualize(self, frame: np.ndarray, metrics: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Add visualization overlays to frame"""
        vis_frame = frame.copy()

        # Draw regions
        for lane_id, region in self.regions.items():
            cv2.polylines(vis_frame, [region.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(vis_frame, lane_id, tuple(region[0].astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw tracks and waiting vehicles
        for track in self.tracks.values():
            if len(track.positions) > 1:
                # Draw track trail
                points = list(track.positions)
                for i in range(1, len(points)):
                    cv2.line(vis_frame, tuple(points[i-1].astype(int)),
                            tuple(points[i].astype(int)), (255, 0, 0), 2)

                # Calculate speed
                if len(track.speeds) > 0:
                    avg_speed_pixels_sec = np.mean(list(track.speeds))
                    speed_kmh = avg_speed_pixels_sec * self.meter_per_pixel * 3.6
                else:
                    speed_kmh = 0.0

                # Get waiting time
                wait_time = track.get_current_wait_time()

                # Draw speed and waiting time
                center = tuple(track.positions[-1].astype(int))
                color = (0, 255, 0) if speed_kmh > 2.0 else (0, 0, 255)  # Green for moving, red for stopped

                cv2.putText(
                    vis_frame,
                    f"ID:{track.track_id} {speed_kmh:.1f}km/h",
                    (center[0] - 50, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                if wait_time > 0:
                    cv2.putText(
                        vis_frame,
                        f"Wait:{wait_time:.1f}s",
                        (center[0] - 50, center[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                # Highlight waiting vehicles
                if track.is_waiting:
                    cv2.circle(vis_frame, center, 10, (0, 0, 255), -1)

        # Add metrics text
        y_offset = 30
        for lane_id, lane_metrics in metrics.items():
            if lane_id != "overall":
                text = f"{lane_id}: Queue={lane_metrics['queue_length']}, Wait={lane_metrics['avg_wait_time']:.1f}s"
                cv2.putText(vis_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25

        return vis_frame


def create_lane_regions(frame_shape: Tuple[int, int], num_lanes: int = 4) -> Dict[str, np.ndarray]:
    """
    Create default lane regions for a typical 4-way intersection
    This is a basic implementation - adjust polygons based on actual camera view
    """
    h, w = frame_shape
    regions = {}

    # Define regions as polygons (adjust these coordinates based on your setup)
    if num_lanes >= 1:
        # South lane (bottom)
        regions["south"] = np.array([
            [0, h*0.7],
            [w*0.4, h*0.7],
            [w*0.4, h],
            [w*0.6, h],
            [w*0.6, h*0.7],
            [w, h*0.7],
            [w, h],
            [0, h]
        ], dtype=np.float32)

    if num_lanes >= 2:
        # East lane (right)
        regions["east"] = np.array([
            [w*0.7, 0],
            [w, 0],
            [w, h*0.4],
            [w*0.7, h*0.4],
            [w*0.7, h*0.6],
            [w, h*0.6],
            [w, h],
            [w*0.7, h]
        ], dtype=np.float32)

    if num_lanes >= 3:
        # West lane (left)
        regions["west"] = np.array([
            [0, 0],
            [w*0.3, 0],
            [w*0.3, h*0.4],
            [0, h*0.4],
            [0, h*0.6],
            [w*0.3, h*0.6],
            [w*0.3, h],
            [0, h]
        ], dtype=np.float32)

    if num_lanes >= 4:
        # North lane (top)
        regions["north"] = np.array([
            [0, 0],
            [w, 0],
            [w, h*0.3],
            [w*0.6, h*0.3],
            [w*0.6, 0],
            [w*0.4, 0],
            [w*0.4, h*0.3],
            [0, h*0.3]
        ], dtype=np.float32)

    return regions


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Queue Length and Waiting Time Estimator")
    parser.add_argument("--video", type=str, default="pi/tests/traffic_jam.mp4", help="Path to video file")
    parser.add_argument("--display", action="store_true", help="Show visualization window")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO model path")
    args = parser.parse_args()

    # Initialize estimator
    regions = create_lane_regions((720, 1280))  # Adjust based on video resolution
    estimator = QueueEstimator(
        model_path=args.model,
        regions=regions,
        conf_threshold=0.3,
        fps=30.0
    )

    # Process video
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get estimates
        metrics = estimator.estimate(frame)

        # Print metrics
        print(f"Overall - Queue: {metrics['overall']['queue_length']}, Avg Wait: {metrics['overall']['avg_wait_time']:.1f}s")

        if args.display:
            # Visualize
            vis_frame = estimator.visualize(frame, metrics)
            cv2.imshow("Queue Estimation", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Small delay for headless operation
            import time
            time.sleep(0.03)

    cap.release()
    if args.display:
        cv2.destroyAllWindows()