"""
ESP32 Camera IP Discovery Tool
Helps you find the ESP32 camera IP address on your network
"""

import socket
import threading
import sys
import time
from pathlib import Path
import yaml

def find_esp32_on_network(network_prefix="192.168.1", timeout=2):
    """
    Scan network to find ESP32 camera
    
    Args:
        network_prefix: Network to scan (e.g., "192.168.1" for 192.168.1.0/24)
        timeout: Timeout per IP in seconds
    
    Returns:
        List of active IPs
    """
    print(f"\n🔍 Scanning network: {network_prefix}.0/24")
    print("   This may take 1-2 minutes...\n")
    
    active_ips = []
    
    def check_ip(ip):
        try:
            # Try to connect to port 81 (ESP32 default)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, 81))
            sock.close()
            
            if result == 0:
                # Port 81 is open, likely ESP32
                print(f"   ✓ Found ESP32 at {ip}:81")
                active_ips.append(ip)
                return True
            
            # Also check port 80 (HTTP)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, 80))
            sock.close()
            
            if result == 0:
                print(f"   ? Found device at {ip}:80 (might be ESP32)")
                active_ips.append(ip)
                return True
                
        except Exception:
            pass
        
        return False
    
    # Scan IPs in parallel
    threads = []
    for i in range(2, 255):
        ip = f"{network_prefix}.{i}"
        t = threading.Thread(target=check_ip, args=(ip,), daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    return active_ips

def test_esp32_stream(ip, port=81):
    """Test if IP has ESP32 camera stream"""
    import cv2
    
    urls = [
        f"http://{ip}:{port}/stream",
        f"http://{ip}:{port}/video_feed",
        f"http://{ip}/stream"
    ]
    
    print(f"\n📹 Testing camera stream at {ip}...")
    
    for url in urls:
        try:
            print(f"   Trying: {url}")
            cap = cv2.VideoCapture(url)
            
            # Try to read a frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"   ✓ Success! Stream available at: {url}")
                return url
        except Exception as e:
            pass
    
    print(f"   ✗ No camera stream found")
    return None

def get_local_ip():
    """Get local IP address of Raspberry Pi"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.1.100"

def main():
    print("\n" + "="*60)
    print("ESP32 Camera IP Discovery Tool")
    print("="*60)
    
    # Get local IP to determine network
    local_ip = get_local_ip()
    network_prefix = ".".join(local_ip.split(".")[:3])
    
    print(f"\nLocal IP: {local_ip}")
    print(f"Network: {network_prefix}.0/24")
    
    # Ask user for confirmation
    response = input("\nScan this network? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return 1
    
    # Scan network
    active_ips = find_esp32_on_network(network_prefix)
    
    if not active_ips:
        print("\n✗ No devices found on network")
        print("\nTroubleshooting:")
        print("1. Make sure ESP32 is powered on")
        print("2. Check that ESP32 is connected to same WiFi network")
        print("3. Try scanning manually: nmap -sn 192.168.1.0/24")
        return 1
    
    print(f"\n✓ Found {len(active_ips)} active device(s)")
    
    # Test each IP for camera stream
    print("\n" + "-"*60)
    print("Testing for camera streams...")
    print("-"*60)
    
    esp32_ips = {}
    for ip in active_ips:
        stream_url = test_esp32_stream(ip)
        if stream_url:
            esp32_ips[ip] = stream_url
    
    # Summary
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    if esp32_ips:
        print(f"\n✓ Found {len(esp32_ips)} ESP32 camera(s):\n")
        for i, (ip, url) in enumerate(esp32_ips.items(), 1):
            print(f"{i}. IP: {ip}")
            print(f"   Stream: {url}")
        
        # Update configuration file
        if len(esp32_ips) == 1:
            ip, url = list(esp32_ips.items())[0]
            
            response = input(f"\nUpdate config with {url}? (y/n): ").lower()
            if response == 'y':
                config_file = Path(__file__).resolve().parent.parent / "configs" / "yolo_config.yaml"
                if config_file.exists():
                    cfg = yaml.safe_load(config_file.read_text()) or {}
                    lanes = cfg.get("lanes", [])
                    if not lanes:
                        print("\n⚠ No lanes configured in configs/yolo_config.yaml")
                    else:
                        lanes[0]["camera_url"] = url
                        cfg["lanes"] = lanes
                        config_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
                        print(f"\n✓ Updated configs/yolo_config.yaml")
                        print(f"  lane_1.camera_url = \"{url}\"")
    else:
        print("\n⚠ No ESP32 cameras found")
        print("But scanning found these IPs (might still be useful):")
        for ip in active_ips:
            print(f"  - {ip}")
    
    print("\n" + "="*60 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
