# ESP32 Traffic Light Integration

## Quick Start

### Start Inference Server

```bash
python3 main.py --mode real
```

Server runs on `http://192.168.x.x:8000`
i.e `http://<Raspberry_Pi_IP>:8000`

### ESP32 Code Example

```cpp
// Every time current phase finishes, ESP32 asks for next decision
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

void setup() {
  WiFi.begin("SSID", "PASSWORD");
  Serial.begin(115200);

  // Wait for WiFi
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
}

void loop() {
  // Request next phase decision from Raspberry Pi
  HTTPClient http;
  String url = "http://192.168.1.X:8000/get_phase";
  http.begin(url);

  int httpCode = http.GET();

  if (httpCode == 200) {
    String response = http.getString();
    Serial.println("Response: " + response);

    // Parse JSON response
    // {"phase": 2, "duration": 30, "yellow_required": true}

    StaticJsonDocument<200> doc;
    deserializeJson(doc, response);

    int phase = doc["phase"];
    int duration = doc["duration"];
    bool yellow = doc["yellow_required"];

    // If yellow required, run yellow light for 3 seconds first
    if (yellow) {
      setYellow();
      delay(3000);
    }

    // Run the phase for its duration
    setPhase(phase);
    delay(duration * 1000);
  } else {
    Serial.println("Error: " + String(httpCode));
  }

  http.end();
}

// Helper functions to control actual traffic lights
void setYellow() {
  // TODO: Implement GPIO control for yellow light
  digitalWrite(YELLOW_PIN, HIGH);
}

void setPhase(int phase) {
  // TODO: Implement GPIO control based on phase
  switch(phase) {
    case 0:  // East-West green
      digitalWrite(EW_GREEN, HIGH);
      digitalWrite(NS_GREEN, LOW);
      break;
    case 2:  // North-South green
      digitalWrite(EW_GREEN, LOW);
      digitalWrite(NS_GREEN, HIGH);
      break;
  }
}
```

## API Endpoints

### GET /get_phase

Returns the next phase decision based on current traffic conditions.

**Field Details:**

- `phase`: Traffic light phase (0=EW green, 2=NS green)
- `duration`: How long to run this phase (seconds, 10-60 with increment of 10)
- `yellow_required`: Whether to show yellow before this phase (boolean)

**Example States:**

```json
{"phase": 0, "duration": 40, "yellow_required": true} 
{"phase": 2, "duration": 30, "yellow_required": false}
```

### GET /health

Health check endpoint.

**Response:**

```json
{ "status": "ok" }
```

## Command Line Options

### Test Mode (Simulated Scenarios, No Cameras)

```bash
python3 main.py --mode test
```

- No camera feeds needed
- Uses hard-coded scenarios for testing
- Perfect for debugging ESP32 code before deployment
- Test scenarios: Heavy lanes, balanced, no congestion

### Real Mode (Live Camera Feeds)

```bash
python3 main.py --mode real
```

- Connects to ESP32 cameras (via configs/yolo_config.yaml)
- Uses YOLO to detect vehicles in real-time
- Feeds live metrics to PPO model
- FastAPI server starts automatically on port 8000

### With Live Camera Display

```bash
python3 main.py --mode real --display
```

- Shows video feeds with detection boxes
- Useful for verifying detection zones (ROI)
- See which vehicles are being tracked
- **Note:** May slow down on Raspberry Pi

### Custom API Server Port

```bash
python3 main.py --mode real --port 8001
```

- If port 8000 is already in use
- ESP32 would then call `http://pi-ip:8001/get_phase`

## Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────┐
│ ESP32 Traffic Light Controller (Hardware)                │
│  - Controls physical traffic lights                      │
│  - Polls /get_phase every phase cycle                    │
└────────────────┬─────────────────────────────────────────┘
                 │ HTTP GET /get_phase
                 │ (JSON response)
                 ↓
┌──────────────────────────────────────────────────────────┐
│ Raspberry Pi: main.py (FastAPI Server port 8000)         │
│ ┌────────────────────────────────────────────────────────┤
│ │ src/engine.py                                          │
│ │ ├─ Loads PPO model (models/model.pt)                   │
│ │ ├─ Reads normalization stats (norm_stats.json)         │
│ │ └─ Runs inference on traffic metrics                   │
│ │                                                        │
│ │ src/api.py (FastAPI endpoints)                         │
│ │ └─ /get_phase ← Returns phase decision                 │
│ │                                                        │
│ │ src/monitor.py (Camera Monitoring Threads)             │
│ │ ├─ Monitors multiple ESP32 cameras                     │
│ │ ├─ Detects vehicles with YOLO                          │
│ │ └─ Calculates: queue_length, avg_wait_time             │
│ │                                                        │
│ │ configs/yolo_config.yaml                               │
│ │ └─ Lane definitions and camera URLs                    │
│ └────────────────────────────────────────────────────────┤
└──────────────────────────────────────────────────────────┘
                 ↓
        Camera Feeds from ESP32s
```

## How It Works Step-by-Step

### Initialization

1. `main.py` starts with `--mode real`
2. Loads PPO model from `models/model.pt`
3. Loads normalization stats from `norm_stats.json`
4. Spawns monitor threads for each lane in `configs/yolo_config.yaml`
5. Starts FastAPI server on port 8000

### Per Phase Cycle (Every 10-60 seconds)

1. ESP32 finishes current phase
2. ESP32 sends HTTP GET to `http://pi-ip:8000/get_phase`
3. `main.py` receives request
4. Queries traffic monitor for latest metrics:
   - `queues`: Number of vehicles in each lane
   - `waits`: Average wait time per lane
5. PPO model runs inference:
   - Input: [queue_1, queue_2, ..., wait_1, wait_2, ..., current_phase]
   - Output: best_phase, best_duration
6. Returns JSON response with yellowto ESP32
7. ESP32 controls traffic lights:
   - (Optional) Show yellow for 3 seconds
   - Show phase_color for duration seconds
8. Repeat from step 2

### Example Request/Response Cycle

**Live Camera Data:**

```
Lane 0 (North): 5 vehicles waiting avg 15.2s
Lane 1 (East):  2 vehicles waiting avg  8.5s
Lane 2 (South): 4 vehicles waiting avg 12.1s
Lane 3 (West):  1 vehicle  waiting avg  3.2s

Queues: [5, 2, 4, 1]
Waits:  [15.2, 8.5, 12.1, 3.2]
```

**PPO Model Inference:**

```
Input: [5, 2, 4, 1, 15.2, 8.5, 12.1, 3.2, current phase]
       (normalized)

Model Decision:
→ Phase 2 is best (handles North & South)
→ Duration 45 seconds (long due to high queue)
→ Yellow required (transitioning from phase 0)
```

**API Response:**

```json
{
  "phase": 2,
  "duration": 45,
  "yellow_required": true
}
```

**ESP32 Execution:**

```
1. Show Yellow for 3 seconds
2. Show North+South Green (Phase 2) for 45 seconds
3. Request new phase again
```

## Configuration

Edit `configs/yolo_config.yaml`:

```yaml
model:
  path: "models/yolo/best.pt" # Primary YOLO model
  fallback_path: "models/yolo/yolo26n.pt" # If primary fails
  conf_threshold: 0.3 # Detection confidence
  vehicle_classes: [1, 2, 3, 5, 7] # COCO: car, motorcycle, bus, truck, train

lanes:
  - id: "lane_0"
    name: "South Lane"
    camera_url: "http://192.168.1.100:81/stream"
    roi:
      y_min: 0.0 # Full height
      y_max: 1.0
      x_min: 0.0 # Full width
      x_max: 1.0

  - id: "lane_1"
    name: "East Lane"
    camera_url: "http://192.168.1.101:81/stream"
    # ... same ROI for all lanes ...

tracking:
  stop_speed_kmh: 3.0 # Speed threshold for "stopped"
  pixel_to_meter: 0.05 # Calibration (adjust per camera)

output:
  file: "traffic_metrics.json" # Auto-generated metrics file
  update_interval: 5 # Update every 5 seconds
  display: false # Set true for video windows
```

## Typical Deployment

### On Raspberry Pi

```bash
# SSH into Pi
ssh pi@192.168.1.X
cd ~/cf-madrl/pi

# One-time setup
chmod +x scripts/setup.sh
./scripts/setup.sh

# Run inference engine (includes API server)
source .venv/bin/activate
python3 main.py --mode real --port 8000

# Or run in background
nohup python3 main.py --mode real > traffic.log 2>&1 &
```

### Auto-Start on Boot

Create systemd service (recommended):

```bash
sudo nano /etc/systemd/system/cf-madrl.service
```

```ini
[Unit]
Description=CF-MADRL Traffic Control API
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

Then enable:

```bash
sudo systemctl enable cf-madrl
sudo systemctl start cf-madrl
sudo systemctl status cf-madrl
```

## Troubleshooting

### ESP32 Can't Connect to /get_phase

```bash
# Check if API server is running
ps aux | grep main.py

# Check port is open
netstat -tlnp | grep 8000

# Test from command line
curl http://192.168.1.X:8000/get_phase
```

### Returns Wrong Phase

1. Check camera feeds are connected:
   ```bash
   python3 tools/find_esp32.py
   ```
2. Verify config has correct camera URLs:
   ```bash
   nano configs/yolo_config.yaml
   ```
3. Check detection is working (enable display):
   ```bash
   python3 main.py --mode real --display
   ```

### Slow Response Times

- Disable `display: true` in yolo_config.yaml
- Reduce `conf_threshold` (0.2 instead of 0.3)
- Use simpler YOLO model (yolo26n.pt)

## See Also

- [Traffic Monitor Documentation](traffic_monitor.md) - Camera monitoring details
- [Setup Commands](../scripts/ALL_COMMANDS.txt) - Complete command reference
- `configs/yolo_config.yaml` - All configuration options with comments
  name: "North Lane"
  camera_url: "http://192.168.1.100:81/stream"
  roi:
  y_min: 0.4
  y_max: 0.9
  x_min: 0.1
  x_max: 0.9

````

## Troubleshooting

**Camera not connecting?**
```bash
python tools/find_esp32.py
````

**ESP32 can't reach server?**

- Check IP address: `hostname -I` on Raspberry Pi
- ESP32 and Pi must be on same network
- Check firewall isn't blocking port 8000

**Seeing wrong traffic decisions?**

- Run with `--display` to see camera feeds
- Adjust ROI zones in configs/yolo_config.yaml
- Lower `conf_threshold` if vehicles not detected
