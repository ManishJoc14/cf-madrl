"""CF-MADRL Traffic Light Inference - Main Entry Point"""

import argparse
from src.engine import TrafficInference
from src.api import run_server, set_engine
from tests.test_mode import run_test_mode


def main():
    parser = argparse.ArgumentParser(
        description="CF-MADRL Traffic Light Inference Engine"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "real"],
        default="real",
        help="Run mode: test (simulated) or real (camera feeds)",
    )
    parser.add_argument(
        "--display", action="store_true", help="Show camera feeds (real mode only)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="API server port (default: 8000)"
    )

    args = parser.parse_args()

    # Initialize engine
    print("Initializing CF-MADRL...")
    try:
        engine = TrafficInference(use_yolo_deploy=(args.mode == "real"))
    except Exception as e:
        print(f"Error: {e}")
        return

    # Enable display if requested
    if args.display and engine.use_yolo_deploy:
        for monitor in engine.lane_monitors.values():
            monitor.config.DISPLAY = True

    # Run requested mode
    if args.mode == "real":
        print("\n" + "=" * 60)
        print("REAL MODE: Camera feeds active")
        print("=" * 60)
        set_engine(engine)
        run_server(port=args.port)
    else:
        print("\n" + "=" * 60)
        print("TEST MODE: Simulated scenarios")
        print("=" * 60)
        run_test_mode(engine)


if __name__ == "__main__":
    main()
