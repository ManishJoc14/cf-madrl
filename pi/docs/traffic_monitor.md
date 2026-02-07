# CF-MADRL Multi-Lane Traffic Monitor

Monitors multiple camera feeds in parallel and provides real-time traffic metrics to the PPO inference engine. Automatically integrated into the main deployment for live edge-based traffic light control on Raspberry Pi.

## 📋 Quick Start

### Option 1: Integrated with Main Inference Engine (Recommended)

The traffic monitor is automatically started when you run the inference engine:

```bash
# Real mode - automatically starts monitor for all configured lanes
python3 main.py --mode real

# Real mode with video display (for debugging)
python3 main.py --mode real --display

# Test mode - simulated scenarios (no cameras needed)
python3 main.py --mode test
```

The monitor runs as background threads and feeds live metrics to the PPO controller.

### Option 2: Standalone Monitor (Legacy)

If you want to run just the monitor without inference:

```bash
source .venv/bin/activate
python3 src/monitor.py
```

This generates `traffic_metrics.json` but doesn't control traffic lights.

## 📂 Folder Structure & Components

```
pi/
├── main.py                     # ← MAIN ENTRY POINT (starts everything)
├── src/
│   ├── engine.py              # Loads PPO model + spawns monitor
│   ├── api.py                 # FastAPI server for ESP32
│   ├── monitor.py             # Traffic monitoring system
│   └── models.py              # Data models
├── configs/
│   └── yolo_config.yaml       # ← EDIT THIS (camera URLs, ROI, etc.)
├── models/
│   ├── model.pt               # PPO policy model
│   └── yolo/
│       ├── best.pt            # Our trained YOLO model
│       └── yolo26n.pt         # Fallback model
├── utils/
│   ├── inference_utils.py     # Normalization helpers
│   └── model.py               # Model loading utilities
├── tools/
│   └── find_esp32.py          # Find camera IPs on network
└── docs/
    ├── ESP32_INTEGRATION.md   # Hardware integration guide
    └── traffic_monitor.md     # This file
```

## ⚙️ Configuration

Edit `configs/yolo_config.yaml` to define your lanes and detection settings:

```yaml
# YOLO Model Settings
model:
  path: "models/yolo/best.pt"              # Primary model (our trained weights)
  fallback_path: "models/yolo/yolo26n.pt"  # Fallback if primary fails
  conf_threshold: 0.3                      # Detection confidence (0.0-1.0)
  vehicle_classes: [1, 2, 3, 5, 7]        # COCO: car, motorcycle, bus, truck, train

# Camera Lanes (update camera_url for your ESP32 IPs)
lanes:
  - id: "lane_0"
    name: "South Lane"
    camera_url: "http://192.168.1.100:81/stream"  # ← UPDATE THIS
    roi:
      y_min: 0.0  # Detection zone: 0.0 = top, 1.0 = bottom
      y_max: 1.0
      x_min: 0.0  # Detection zone: 0.0 = left, 1.0 = right
      x_max: 1.0
  
  - id: "lane_1"
    name: "East Lane"
    camera_url: "http://192.168.1.101:81/stream"  # ← UPDATE THIS
    roi:
      y_min: 0.0
      y_max: 1.0
      x_min: 0.0
      x_max: 1.0
  
  # Add more lanes as needed...

# Vehicle Tracking
tracking:
  stop_speed_kmh: 3.0           # Speed threshold to classify as "stopped"
  pixel_to_meter: 0.05          # Calibration factor (adjust per camera height/angle)

# Output
output:
  file: "traffic_metrics.json"  # Where metrics are saved
  update_interval: 5            # Update every 5 seconds
  display: false                # Set true to show video windows
```

### Finding ESP32 Camera IPs

```bash
source .venv/bin/activate
python3 tools/find_esp32.py
```

Output will show available ESP32 cameras on your network.

## 🎯 How Monitor Works

```
ESP32 Camera Feeds
       ↓
[Monitor Thread 1] → Lane 0: Detects vehicles → Queue length: 3
[Monitor Thread 2] → Lane 1: Detects vehicles → Queue length: 5
[Monitor Thread 3] → Lane 2: Detects vehicles → Queue length: 1
[Monitor Thread 4] → Lane 3: Detects vehicles → Queue length: 2
       ↓
Traffic Metrics Update (every ~5 seconds)
       ↓
PPO Engine reads metrics → Decides best phase
       ↓
ESP32 receives phase decision via /get_phase API
```

### Per-Lane Monitoring

For each configured lane, the monitor:

1. **Connects** to ESP32 camera stream
2. **Detects** vehicles using YOLO model
3. **Tracks** vehicle movement and speed
4. **Calculates**:
   - Queue length (stopped vehicles in ROI)
   - Average wait time per vehicle
   - Total vehicles in frame
   - Processing FPS

Example output:
```
✓ South Lane connected: http://192.168.1.100:81/stream
  → YOLO detecting: confidence threshold 0.3
  → Detected vehicles: 8
  → Queue length: 3
  → Avg wait time: 12.5 seconds
  → FPS: 15.2

✓ East Lane connected: http://192.168.1.101:81/stream
  → Detected vehicles: 5
  → Queue length: 5 (heavy congestion!)
  → Avg wait time: 28.3 seconds
  → FPS: 14.8
```

## 📊 Metrics Output

### Real-Time Integration (Recommended)

When running with `main.py`, the engine automatically reads metrics via:

```python
from src.engine import TrafficInference

engine = TrafficInference(use_yolo_deploy=True)

# Get latest metrics from all lanes
queues, waits = engine.get_real_metrics()
# queues = [3, 5, 1, 2]       (vehicles waiting in each lane)
# waits = [12.5, 28.3, 5.0, 8.1]  (average wait time per lane)

# PPO model makes decision based on this data
phase, duration, yellow = engine.get_action(queues, waits, current_phase=0)
# → phase: 2 (prioritize East lane with 5 vehicles)
# → duration: 45 (long phase due to high congestion)
# → yellow: true (need yellow transition)
```

No manual JSON file parsing needed!

### Legacy JSON File Output

The monitor also writes to `traffic_metrics.json` every 5 seconds:

```json
{
  "timestamp": 1707049324.5,
  "lanes": {
    "lane_0": {
      "queue_length": 3,
      "avg_wait_time": 12.5,
      "total_vehicles": 8,
      "fps": 15.2
    },
    "lane_1": {
      "queue_length": 5,
      "avg_wait_time": 28.3,
      "total_vehicles": 5,
      "fps": 14.8
    },
    "lane_2": {
      "queue_length": 1,
      "avg_wait_time": 5.0,
      "total_vehicles": 3,
      "fps": 16.1
    },
    "lane_3": {
      "queue_length": 2,
      "avg_wait_time": 8.1,
      "total_vehicles": 4,
      "fps": 15.5
    }
  },
  "total_queue_length": 11,
  "total_vehicles": 20
}
```

If you want to read this manually:

```python
import json

with open('traffic_metrics.json', 'r') as f:
    metrics = json.load(f)

# Get data for individual lanes
south_queue = metrics['lanes']['lane_0']['queue_length']
east_queue = metrics['lanes']['lane_1']['queue_length']

print(f"South: {south_queue} vehicles waiting")
print(f"East: {east_queue} vehicles waiting")
```

## 🚀 Run Modes

### Mode 1: Integrated Inference (Recommended)

```bash
# Real mode with live camera feeds
python3 main.py --mode real

# Show video windows for debugging
python3 main.py --mode real --display

# Custom port (if 8000 is in use)
python3 main.py --mode real --port 8001
```

**What happens:**
- Monitor connects to all configured cameras
- YOLO detects vehicles in real-time
- Metrics fed to PPO model
- FastAPI server handles /get_phase requests from ESP32
- Runs until you press Ctrl+C

### Mode 2: Test Mode (No Cameras)

```bash
python3 main.py --mode test
```

**What happens:**
- Uses simulated traffic scenarios
- No camera connections needed
- Good for testing before deployment
- Useful for debugging ESP32 code

### Mode 3: Standalone Monitor (Legacy)

```bash
python3 src/monitor.py
```

**What happens:**
- Only runs the camera monitoring
- Writes metrics to traffic_metrics.json
- Doesn't run PPO model or API server
- Useful if PPO runs on a different machine

With display:
```bash
python3 src/monitor.py --display
```

## 🔍 Troubleshooting

### Camera Connection Fails

```bash
# Find available ESP32s on network
python3 tools/find_esp32.py

# Update config with correct IP
nano configs/yolo_config.yaml
```

### Vehicle Detection Not Working

1. **Check detection confidence:**
   ```yaml
   # In configs/yolo_config.yaml
   model:
     conf_threshold: 0.3  # Lower for more detections (0.1-0.5 range)
   ```

2. **Visualize detection zones:**
   ```bash
   python3 main.py --mode real --display
   ```
   This shows video with detection boxes. Adjust ROI if needed:
   ```yaml
   roi:
     y_min: 0.0  # 0.0 = top
     y_max: 1.0  # 1.0 = bottom
     x_min: 0.0  # 0.0 = left  
     x_max: 1.0  # 1.0 = right
   ```

3. **Check model files exist:**
   ```bash
   ls -la models/yolo/
   # Should show: best.pt and yolo26n.pt
   ```

### Low Frame Rate / Slow Processing

**On Raspberry Pi:**
- Reduce `conf_threshold` (e.g., 0.2)
- Disable `display: true`
- Use fewer lanes (2 instead of 4)
- Reduce image resolution (if ESP32 supports)

```yaml
output:
  display: false           # Don't show windows
  
model:
  conf_threshold: 0.2     # Lower threshold = faster
```

### Monitor Not Updating Metrics

1. Check monitor is running:
   ```bash
   ps aux | grep main.py
   ```

2. Check update interval:
   ```yaml
   output:
     update_interval: 5  # Update every 5 seconds
   ```

3. Check logs:
   ```bash
   tail -f traffic_metrics.json  # Watch live updates
   ```

### Monitor Falls Back to Test Mode

- Verify all camera URLs are correct
- Ensure ESP32 cameras are powered on and transmitting
- Make sure YOLO models exist: `ls models/yolo/best.pt`

## 📈 Performance Optimization

### For Raspberry Pi 4

**Standard (2-3 lanes):**
```yaml
model:
  conf_threshold: 0.3

output:
  display: false
  update_interval: 5
```

**Light Load (fast response):**
```yaml
model:
  conf_threshold: 0.2

output:
  display: false
  update_interval: 2
```

**Heavy Load (more accurate):**
```yaml
model:
  conf_threshold: 0.4
  fallback_path: "models/yolo/yolo26n.pt"  # Use lighter model first

output:
  display: false
  update_interval: 10
```

## 🔄 Auto-Start on Boot

### Option 1: Systemd Service (Recommended)

```bash
sudo nano /etc/systemd/system/cf-madrl.service
```

```ini
[Unit]
Description=CF-MADRL Traffic Control (with Monitor)
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/cf-madrl/pi
ExecStart=/home/pi/cf-madrl/pi/.venv/bin/python3 main.py --mode real --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable cf-madrl
sudo systemctl start cf-madrl
sudo systemctl status cf-madrl
```

### Option 2: Crontab

```bash
crontab -e

# Add this line:
@reboot sleep 30 && cd /home/pi/cf-madrl/pi && /home/pi/cf-madrl/pi/.venv/bin/python3 main.py --mode real
```

## 📚 Related Documentation

- [ESP32 Integration Guide](ESP32_INTEGRATION.md) - Hardware control and API details
- [Setup Commands](../scripts/ALL_COMMANDS.txt) - Complete installation steps
- `configs/yolo_config.yaml` - All configuration options
- `main.py` - Main entry point code
- `src/engine.py` - Inference engine with monitor integration
- `src/monitor.py` - Monitor implementation details
