# CF-MADRL Multi-Lane Traffic Monitor

Monitors multiple camera feeds in parallel and provides traffic metrics for PPO traffic light controller.

## 📋 Quick Setup

### 1. Transfer to Raspberry Pi
```bash
scp -r yolo-deploy/ pi@192.168.1.XXX:~/cf-madrl/
```

### 2. SSH and Install
```bash
ssh pi@<PI_IP>
cd ~/cf-madrl/yolo-deploy
chmod +x setup.sh && ./setup.sh
```

### 3. Configure Lanes
```bash
nano config.yaml
# Update camera_url for each lane with ESP32 IPs
```

### 4. Run
```bash
source .venv/bin/activate
python3 monitor.py
```

## ⚙️ Multi-Lane Configuration

Edit `config.yaml` to define your lanes:

```yaml
model:
  path: "best.pt"              # Our trained model
  fallback_path: "yolo26n.pt"  # Fallback if primary fails to detect

lanes:
  - id: "lane_1"
    name: "North Lane"
    camera_url: "http://192.168.1.100:81/stream"
    roi:
      y_min: 0.4  # Detection zone
      y_max: 0.9
      x_min: 0.1
      x_max: 0.9
```

**How it works:** If the primary model (best.pt) doesn't detect vehicles, the fallback model (yolov8n.pt) automatically takes over to ensure continuous monitoring.

## 🎯 Output Format

`traffic_metrics.json` updated every 5 seconds:

```json
{
  "timestamp": 1707049324.5,
  "lanes": {
    "lane_1": {"queue_length": 3, "avg_wait_time": 12.5, "total_vehicles": 8},
    "lane_2": {"queue_length": 2, "avg_wait_time": 8.3, "total_vehicles": 5},
    "lane_3": {"queue_length": 1, "avg_wait_time": 5.2, "total_vehicles": 3},
    "lane_4": {"queue_length": 0, "avg_wait_time": 0.0, "total_vehicles": 2}
  },
  "total_queue_length": 6,
  "avg_wait_time": 8.67,
  "total_vehicles": 18
}
```

## 🤖 PPO Integration

```python
import json

with open('traffic_metrics.json', 'r') as f:
    metrics = json.load(f)

# Individual lanes
north_queue = metrics['lanes']['lane_1']['queue_length']
north_wait = metrics['lanes']['lane_1']['avg_wait_time']

south_queue = metrics['lanes']['lane_2']['queue_length']
south_wait = metrics['lanes']['lane_2']['avg_wait_time']

east_queue = metrics['lanes']['lane_3']['queue_length']
east_wait = metrics['lanes']['lane_3']['avg_wait_time']

west_queue = metrics['lanes']['lane_4']['queue_length']
west_wait = metrics['lanes']['lane_4']['avg_wait_time']

# Build state vector for PPO
state = [[ north_queue, south_queue, east_queue, west_queue], 
         [north_wait, south_wait, east_wait, west_wait], 
         [current_phase]]

action = ppo_actor.predict(state)
```

## 🚀 Run Modes

```bash
# Background mode (no display)
python3 monitor.py

# With video windows (testing)
python3 monitor.py --display

# Custom config
python3 monitor.py --config my_config.yaml

# Stop: Ctrl+C
```

## 🔍 Troubleshooting

**No camera connection:**
```bash
python3 find_esp32.py  # Find ESP32 IPs
```

**Check logs:**
Monitor prints status for each lane every 5 seconds

**Test single lane:**
Edit config.yaml to include only one lane entry

**ROI not detecting:**
Set `display: true` in config.yaml to see detection zone

## 📚 Full Documentation

- `ALL_COMMANDS.txt` - Complete command reference
- `config.yaml` - All settings with comments

## 🔄 Auto-Start on Boot

```bash
crontab -e
# Add this line:
@reboot sleep 30 && cd /home/pi/cf-madrl/yolo-deploy && /home/pi/cf-madrl/yolo-deploy/.venv/bin/python3 monitor.py
```
