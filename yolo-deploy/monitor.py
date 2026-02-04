#!/usr/bin/env python3
"""
CF-MADRL Multi-Lane Traffic Monitor
Monitors multiple camera feeds and provides real-time traffic metrics to PPO controller
"""

import cv2
import numpy as np
import time
import json
import yaml
import sys
import threading
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# ==================================================================================================================
# CONFIGURATION
# ==================================================================================================================

class Config:
    """Load all settings from config.yaml"""
    
    def __init__(self, config_file="config.yaml"):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Model
        m = cfg.get('model', {})
        self.MODEL_PATH = m.get('path', 'best.pt')
        self.FALLBACK_MODEL = m.get('fallback_path', 'yolo26n.pt')  # Fallback if primary fails
        self.CONF_THRESHOLD = m.get('conf_threshold', 0.3)
        self.VEHICLE_CLASSES = m.get('vehicle_classes', [1, 2, 3, 5, 7])
        
        # Lanes (multiple cameras) - supports both new and old format
        self.LANES = cfg.get('lanes', [])
        if not self.LANES:
            # Backward compatibility - single camera
            c = cfg.get('camera', {})
            r = cfg.get('roi', {})
            self.LANES = [{
                'id': 'lane_1',
                'name': 'Lane 1',
                'camera_url': c.get('url', 'http://192.168.1.100:81/stream'),
                'roi': {
                    'y_min': r.get('y_min', 0.4),
                    'y_max': r.get('y_max', 0.9),
                    'x_min': r.get('x_min', 0.1),
                    'x_max': r.get('x_max', 0.9)
                }
            }]
        
        # Tracking
        t = cfg.get('tracking', {})
        self.STOP_SPEED_KMH = t.get('stop_speed_kmh', 3.0)
        self.PIXEL_TO_METER = t.get('pixel_to_meter', 0.05)
        
        # Output
        o = cfg.get('output', {})
        self.DISPLAY = o.get('display', False)
        self.OUTPUT_FILE = o.get('file', 'traffic_metrics.json')
        self.UPDATE_INTERVAL = o.get('update_interval', 5)
        
        print(f"✓ Config loaded: {config_file}")
        print(f"✓ Monitoring {len(self.LANES)} lane(s)")

# ==================================================================================================================
# VEHICLE TRACKER
# ==================================================================================================================

class VehicleTracker:
    """Track individual vehicle position, speed, and waiting time"""
    
    def __init__(self, track_id):
        self.id = track_id
        self.positions = deque(maxlen=10)
        self.speeds = deque(maxlen=10)
        self.wait_start = None
        self.in_roi = False
    
    def update(self, cx, cy, current_time, fps, pixel_to_meter):
        """Update position and calculate speed"""
        self.positions.append((cx, cy))
        
        # Calculate speed
        speed_kmh = 0.0
        if len(self.positions) > 1:
            prev_cx, prev_cy = self.positions[-2]
            dist_px = np.hypot(cx - prev_cx, cy - prev_cy)
            speed_kmh = (dist_px * pixel_to_meter * fps) * 3.6
        
        self.speeds.append(speed_kmh)
        return speed_kmh
    
    def is_stopped(self, threshold):
        """Check if vehicle is stopped based on speed threshold"""
        if len(self.speeds) == 0:
            return False
        return np.mean(list(self.speeds)[-3:]) < threshold
    
    def get_wait_time(self):
        """Get current waiting time"""
        if self.wait_start:
            return time.time() - self.wait_start
        return 0.0

# ==================================================================================================================
# LANE MONITOR (Single Camera Feed)
# ==================================================================================================================

class LaneMonitor:
    """Monitors a single lane/camera feed"""
    def __init__(self, lane_config, primary_model, fallback_model, config):
        self.lane_id = lane_config['id']
        self.lane_name = lane_config.get('name', self.lane_id)
        self.camera_url = lane_config['camera_url']
        self.roi_config = lane_config['roi']
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.config = config
        self.config = config
        
        # Connect to camera
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot connect to {self.lane_name}: {self.camera_url}")
        
        self.tracks = {}
        self.roi_bounds = None
        self.running = True
        self.frame_count = 0
        self.latest_metrics = {
            'queue_length': 0,
            'avg_wait_time': 0.0,
            'total_vehicles': 0
        }
        
        print(f"✓ {self.lane_name} connected: {self.camera_url}")
    
    def _set_roi(self, height, width):
        """Set detection zone boundaries"""
        roi = self.roi_config
        self.roi_bounds = (
            int(width * roi['x_min']),
            int(height * roi['y_min']),
            int(width * roi['x_max']),
            int(height * roi['y_max'])
        )
    
    def _is_in_roi(self, cx, cy):
        """Check if point is inside ROI"""
        if not self.roi_bounds:
            return False
        x_min, y_min, x_max, y_max = self.roi_bounds
        return x_min < cx < x_max and y_min < cy < y_max
    def _process_frame(self, frame):
        """Run YOLO detection with fallback if needed"""
        # Try primary model
        results = self.primary_model.track(
            frame,
            conf=self.config.CONF_THRESHOLD,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )
        
        # Check if primary model detected vehicles
        has_detections = False
        if results and results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls[0]) in self.config.VEHICLE_CLASSES:
                    has_detections = True
                    break
        
        # Fallback to secondary model if nothing detected
        if not has_detections:
            results = self.fallback_model.track(
                frame,
                conf=self.config.CONF_THRESHOLD,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False
            )
        
        detections = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, yolo_id, cls_id in zip(boxes, ids, clss):
                if cls_id not in self.config.VEHICLE_CLASSES:
                    continue
                
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int(y2)  # Bottom center
                
                # Update or create track
                if yolo_id not in self.tracks:
                    self.tracks[yolo_id] = VehicleTracker(yolo_id)
                
                track = self.tracks[yolo_id]
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                speed = track.update(cx, cy, time.time(), fps, self.config.PIXEL_TO_METER)
                
                # Update ROI status
                track.in_roi = self._is_in_roi(cx, cy)
                
                # Update waiting time
                if track.in_roi and track.is_stopped(self.config.STOP_SPEED_KMH):
                    if track.wait_start is None:
                        track.wait_start = time.time()
                else:
                    track.wait_start = None
                
                detections.append({
                    'box': box,
                    'id': yolo_id,
                    'cx': cx,
                    'cy': cy,
                    'speed': speed,
                    'wait': track.get_wait_time()
                })
        
        return detections
    
    def _draw_frame(self, frame, detections):
        """Annotate frame with detections"""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            speed = det['speed']
            wait = det['wait']
            
            color = (0, 255, 0) if speed > self.config.STOP_SPEED_KMH else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{det['id']} {speed:.1f}km/h", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if wait > 0:
                cv2.putText(frame, f"Wait:{wait:.1f}s", (x1, y2+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Draw ROI
        if self.roi_bounds:
            x_min, y_min, x_max, y_max = self.roi_bounds
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
        
        # Draw lane name and metrics
        queue_len, avg_wait = self._get_metrics()
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(frame, self.lane_name, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Queue: {queue_len}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg Wait: {avg_wait:.1f}s", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _get_metrics(self):
        """Calculate queue length and average wait time"""
        queue_length = 0
        total_wait = 0.0
        
        for track in self.tracks.values():
            if track.in_roi and track.is_stopped(self.config.STOP_SPEED_KMH):
                queue_length += 1
                total_wait += track.get_wait_time()
        
        avg_wait = total_wait / queue_length if queue_length > 0 else 0.0
        self.latest_metrics = {
            'queue_length': queue_length,
            'avg_wait_time': avg_wait,
            'total_vehicles': len(self.tracks)
        }
        return queue_length, avg_wait
    
    def process_loop(self):
        """Main processing loop for this lane"""
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Set ROI on first frame
                if self.roi_bounds is None:
                    h, w = frame.shape[:2]
                    self._set_roi(h, w)
                
                # Process frame
                detections = self._process_frame(frame)
                
                # Update metrics
                self._get_metrics()
                
                # Display if enabled
                if self.config.DISPLAY:
                    frame = self._draw_frame(frame, detections)
                    cv2.imshow(f"{self.lane_name}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
        
        finally:
            self.cap.release()
            if self.config.DISPLAY:
                cv2.destroyWindow(self.lane_name)
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

# ==================================================================================================================
# MULTI-LANE COORDINATOR
# ==================================================================================================================

class MultiLaneMonitor:
    """Coordinates multiple lane monitors"""
    
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.MODEL_PATH)
        self.running = True
        
        # Create lane monitors
        self.lanes = []
        for lane_cfg in config.LANES:
            try:
                lane = LaneMonitor(lane_cfg, self.model, config)
                self.lanes.append(lane)
            except Exception as e:
                print(f"[WARNING] Failed to initialize {lane_cfg.get('name', lane_cfg['id'])}: {e}")
        
        if not self.lanes:
            raise RuntimeError("No lanes successfully initialized")
    
    def _save_metrics(self):
        """Aggregate and save metrics from all lanes"""
        all_metrics = {
            'timestamp': time.time(),
            'lanes': {}
        }
        
        total_queue = 0
        total_wait_time = 0.0
        total_vehicles = 0
        
        for lane in self.lanes:
            lane_data = {
                'queue_length': lane.latest_metrics['queue_length'],
                'avg_wait_time': lane.latest_metrics['avg_wait_time'],
                'total_vehicles': lane.latest_metrics['total_vehicles']
            }
            all_metrics['lanes'][lane.lane_id] = lane_data
            
            total_queue += lane_data['queue_length']
            total_wait_time += lane_data['avg_wait_time']
            total_vehicles += lane_data['total_vehicles']
        
        # Add aggregated totals
        all_metrics['total_queue_length'] = total_queue
        all_metrics['avg_wait_time'] = total_wait_time / len(self.lanes) if self.lanes else 0.0
        all_metrics['total_vehicles'] = total_vehicles
        
        # Save to file
        with open(self.config.OUTPUT_FILE, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def run(self):
        """Start monitoring all lanes"""
        print("\n" + "="*70)
        print("CF-MADRL MULTI-LANE TRAFFIC MONITOR - RUNNING")
        print("="*70)
        print(f"Lanes: {len(self.lanes)}")
        print(f"Output: {self.config.OUTPUT_FILE}")
        print(f"Update interval: {self.config.UPDATE_INTERVAL}s")
        print("Press Ctrl+C to stop\n")
        
        # Start each lane in its own thread
        threads = []
        for lane in self.lanes:
            thread = threading.Thread(target=lane.process_loop, daemon=True)
            thread.start()
            threads.append(thread)
        
        try:
            # Main output loop
            while self.running:
                time.sleep(self.config.UPDATE_INTERVAL)
                
                # Save metrics
                self._save_metrics()
                
                # Print status
                for lane in self.lanes:
                    metrics = lane.latest_metrics
                    print(f"[{lane.lane_name}] Queue: {metrics['queue_length']} | "
                          f"Wait: {metrics['avg_wait_time']:.1f}s | "
                          f"Vehicles: {metrics['total_vehicles']}")
                print("-" * 70)
        
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")
        finally:
            # Stop all lanes
            self.running = False
            for lane in self.lanes:
                lane.stop()
            
            # Wait for threads
            for thread in threads:
                thread.join(timeout=2)
            
            if self.config.DISPLAY:
                cv2.destroyAllWindows()
            print("✓ All lanes stopped")

# ==================================================================================================================
# MAIN
# ==================================================================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CF-MADRL Multi-Lane Traffic Monitor")
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--display', action='store_true', help='Show video output')
    args = parser.parse_args()
    
    # Load config
    try:
        config = Config(args.config)
        if args.display:
            config.DISPLAY = True
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {args.config}")
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return 1
    
    # Run monitor
    try:
        monitor = MultiLaneMonitor(config)
        monitor.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
